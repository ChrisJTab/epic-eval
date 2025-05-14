//
// Created by Shujian Qian on 2023-10-25.
/*
This file is responsible for storing and managing records and versions in Epic’s TPC-C transaction processing system
*/
//

#ifndef STORAGE_H
#define STORAGE_H

#include <cstdint>
#include <xmmintrin.h>

#include <execution_planner.h>
#include <util_opt.h>
#include <util_arch.h>

namespace epic {

template<typename ValueType>
struct Record
{
    uint32_t version1 = 0, version2 = 0; // epoch timestamps of the two versions
    ValueType value1, value2; // Data of the two versions
} __attribute__((aligned(kDeviceCacheLineSize))); // forces each record to start on a new cache line

/* make sure Version is properly aligned for GPU atomic operations */
// 128 bytes is the size of a cache line on the GPU
// This avoids false sharing, where two threads access the same cache line
static_assert(sizeof(Record<int>) == kDeviceCacheLineSize);

/* make sure that the two versions are adjacent in memory, so they can be atomically read with 64bit instr */
static_assert(offsetof(Record<int>, version2) - offsetof(Record<int>, version1) == sizeof(uint32_t));

template<typename ValueType>
struct Version // Simpler version of Record, maybe used when multi-versioning is not needed
{
    uint32_t version = 0;
    ValueType value;
} __attribute__((aligned(kDeviceCacheLineSize)));

/* make sure Version is properly aligned for GPU atomic operations */
static_assert(sizeof(Version<int>) == kDeviceCacheLineSize);

// The logic here is synonomous to the logic in Algorithm 1 from the paper
// This function is Function  ReadFromTable(rec_id,read_loc): from the paper
template<typename ValueType>
EPIC_FORCE_INLINE void readFromTable(Record<ValueType> *record, Version<ValueType> *version, uint32_t record_id,
    uint32_t read_loc, uint32_t epoch, ValueType *result)
{
    /* record a read */
    /*
    All of this if statement is equivalent to the following:
    if read_loc==prevVer then:
        read_ver=prevVer
    */
    if (read_loc == loc_record_a) // loc_record_a is prevVer
    {
        /* reading the version from previous epoch, no syncronization needed */
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        ValueType *value_to_read = nullptr;
        if (version1 == epoch)
        {
            /*
            If the version1 is written in this epoch, then it will be the currVer of this epoch
            Since we are in the case of log_record_a(which corresponds to prevVer), we need to read the other version
            i.e. value2 is the correct prevVer
            */
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
        }
        else if (version2 == epoch)
        {
            /*
            If the version2 is written in this epoch, then it will be the currVer of this epoch
            Since we are in the case of log_record_a(which corresponds to prevVer), we need to read the other version
            i.e. value1 is the correct prevVer
            */
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
        }
        else if (version1 < version2)
        {
            /*
            if neither version1 nor version2 is written in this epoch, then we want the latest version before this epoch
            */
            /* version2 is the latest version before this epoch (record_a) */
            value_to_read = &record[record_id].value2;
        }
        else
        {
            /* version1 is the latest version before this epoch (record_a) */
            value_to_read = &record[record_id].value1;
        }
        memcpy(result, value_to_read, sizeof(ValueType));
        return;
    }

    /* record b read */
    /*
    All of this if statement is equivalent to the following:
    else if read_loc==currVer then:
        read_ver=currVer
    */
    if (read_loc == loc_record_b)
    {
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        ValueType *value_to_read = nullptr;
        if (version1 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
        }
        else if (version2 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
        }
        else if (version1 < version2)
        {
            /* version1 will be written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
            // Wait until the version1 is written in this epoch
            // Note that we are reading CurrVer here, so it might not be written yet
            while (__atomic_load_n(&record[record_id].version1, __ATOMIC_SEQ_CST) != epoch)
            {
                _mm_pause();
            }
        }
        else
        {
            /* version2 will be written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
            // Wait until the version2 is written in this epoch
            while (__atomic_load_n(&record[record_id].version2, __ATOMIC_SEQ_CST) != epoch)
            {
                _mm_pause();
            }
        }
        memcpy(result, value_to_read, sizeof(ValueType));
        return;
    }

    /* version read */
    /*
    The following is equivalent to the following:
    else:
        read_ver=tempVers[read_loc.index]
    I assume this means that the passed in version is in an array of temporary versions 
    We wait until the version is written in this epoch regardless of the location
    */
    while (__atomic_load_n(&version[read_loc].version, __ATOMIC_SEQ_CST) != epoch)
    {
        _mm_pause();
    }
    memcpy(result, &version[read_loc].value, sizeof(ValueType));
}

/*
This is Function  writeToTable(rec_id,write_loc,data): from the paper
*/
template<typename ValueType>
EPIC_FORCE_INLINE void writeToTable(Record<ValueType> *record, Version<ValueType> *version, uint32_t record_id,
    uint32_t write_loc, uint32_t epoch, ValueType *source)
{
    /*
    This if statement is equivalent to the following:
    if write_loc==currVer then 
        write_ver=currVer
    */
    if (write_loc == loc_record_b)
    {
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        /* TODO: I don't think atomic read is required here */
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        if (version1 < version2)
        {
            /*
            We want to write to the currVer.
            Epic ensures that the version with a larger epoch ID contains the more up-to-date value and should be used as prevVer
            This means that since version 2 has the larger epoch ID, it would be used as prevVer, so v1 is the currVer
            */
            /* version2 is the latest version before this epoch (record_a) */
            memcpy(&record[record_id].value1, source, sizeof(ValueType));
            __atomic_store_n(&record[record_id].version1, epoch, __ATOMIC_SEQ_CST);
        }
        else
        {
            /* version1 is the latest version before this epoch (record_a) */
            memcpy(&record[record_id].value2, source, sizeof(ValueType));
            __atomic_store_n(&record[record_id].version2, epoch, __ATOMIC_SEQ_CST);
        }
        return;
    }

    /* version write */
//    memcpy(&version[write_loc].value, source, sizeof(ValueType));
    /*
    else //tempVerwrite 
        write_ver=tempVers[write_loc.index]
    */
    __atomic_store_n(&version[write_loc].version, epoch, __ATOMIC_SEQ_CST);
}

} // namespace epic

#endif // STORAGE_H
