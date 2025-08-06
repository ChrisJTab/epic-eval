//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_STORAGE_H
#define EPIC_BENCHMARKS_YCSB_STORAGE_H

#include <variant>
#include <storage.h>
#include <benchmarks/ycsb_table.h>

namespace epic::ycsb {

enum class RecordTier : uint8_t { GPU, CPU };

template <typename RecT, typename VerT>
struct RecordSlice {
    RecordTier  tier;        // GPU or CPU
    RecT*       rec_base;    // base pointer for Record<…>
    VerT*       ver_base;    // base pointer for Version<…>
    uint64_t    count;       // #records in this slice
};

template <typename RecT, typename VerT>
struct TwoTierLayout {
    RecordSlice<RecT,VerT> gpu{};
    RecordSlice<RecT,VerT> cpu{};
};

using YcsbVersions = Version<YcsbValue>;
using YcsbRecords = Record<YcsbValue>;

using YcsbFieldVersions = Version<YcsbFieldValue>;
using YcsbFieldRecords = Record<YcsbFieldValue>;

using YcsbVersionArrType = std::variant<YcsbVersions *, YcsbFieldVersions *>;
using YcsbRecordArrType = std::variant<YcsbRecords *, YcsbFieldRecords *>;

using YcsbSplitLayout = TwoTierLayout<YcsbFieldRecords, YcsbFieldVersions>; // for split-field YCSB
using YcsbFullLayout = TwoTierLayout<YcsbRecords, YcsbVersions>;

}

#endif // EPIC_BENCHMARKS_YCSB_STORAGE_H
