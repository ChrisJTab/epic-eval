//
// Created by Shujian Qian on 2023-08-08.
//
/*
This file is the main point of entry for the EPIC benchmarking suite. 
It parses the command line arguments and runs the benchmark.
*/

#include <iostream>
#include <memory>
#include <getopt.h>

#include "gpu_execution_planner.h"
#include "gpu_allocator.h"
#include <benchmarks/benchmark.h>
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"

#include "benchmarks/tpcc.h"
#include "gacco/benchmarks/tpcc.h"
#include "gacco/benchmarks/ycsb.h"

/*
This file parses the following command line arguments:
--benchmark: the name of the benchmark to run
--database: the name of the database to use
--num_warehouses: the number of warehouses to use
--skew_factor: the skew factor to use
--fullread: whether to use full record read
--cpu_exec_num_threads: the number of CPU execution threads to use
--num_epochs: the number of epochs to use
--num_txns: the number of transactions to use
--split_fields: whether to split fields
--commutative_ops: whether to use commutative operations
--num_records: the number of records to use
--exec_device: the execution device to use

These command line arguments are taken from the file run_experiments.py

long_options is an array of struct option that contains the long options that the program can take
optstring is a string that contains the short options that the program can take, each letter corresponds to a long option

*/
static constexpr struct option long_options[] = {{"benchmark", required_argument, nullptr, 'b'},
    {"database", required_argument, nullptr, 'd'}, {"num_warehouses", required_argument, nullptr, 'w'},
    {"skew_factor", required_argument, nullptr, 'a'}, {"fullread", required_argument, nullptr, 'r'},
    {"cpu_exec_num_threads", required_argument, nullptr, 'c'}, {"num_epochs", required_argument, nullptr, 'e'},
    {"num_txns", required_argument, nullptr, 's'}, {"split_fields", required_argument, nullptr, 'f'},
    {"commutative_ops", required_argument, nullptr, 'm'}, {"num_records", required_argument, nullptr, 'n'},
    {"exec_device", required_argument, nullptr, 'x'},
    {nullptr, 0, nullptr, 0}}; 

static char optstring[] = "b:d:w:a:r:c:e:s:f:m:n:x:";

int main(int argc, char **argv)
{

    epic::tpcc::TpccConfig tpcc_config; // this is the configuration for the Tpcc benchmark
    epic::ycsb::YcsbConfig ycsb_config; // this is the configuration for the Ycsb benchmark

    int retval = 0;
    char *end_char = nullptr; 
    std::string bench = "tpcc"; // the default benchmark is Tpcc
    std::string db = "epic"; // the default database is epic
    bool commutative_ops = false; // whether to use commutative operations
    /*
    getopt_long is a function that parses the command line arguments and comes from the getopt.h library
    it returns the next option in the command line arguments
    */
    while ((retval = getopt_long(argc, argv, optstring, long_options, nullptr)) != -1) // this loop parses the command line arguments
    {
        switch (retval)
        {
        case 'b': // this case is for the benchmark name
            bench = std::string(optarg);
            if (bench == "ycsba")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 50, 0, 0}; // the transaction mix for the Ycsb benchmark
            }
            else if (bench == "ycsbb")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {95, 5, 0, 0};
            }
            else if (bench == "ycsbc")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {100, 0, 0, 0};
            }
            else if (bench == "ycsbf")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 0, 50, 0};
            }
            else if (bench == "tpccn")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {100, 0, 0, 0, 0};
            }
            else if (bench == "tpccp")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {0, 100, 0, 0, 0};
            }
            else if (bench == "tpcc")
            {
                tpcc_config.txn_mix = {50, 50, 0, 0, 0};
            }
            else if (bench == "tpccfull")
            {
                bench = "tpcc";
                tpcc_config.txn_mix = {45, 43, 4, 4, 4};
            }
            else
            {

                throw std::runtime_error("Invalid benchmark name");
            }
            break;
        case 'd': // this case is for the database name
            db = std::string(optarg);
            if (db != "epic" && db != "gacco")
            {
                throw std::runtime_error("Invalid database name");
            }
            break;
        case 'w': // this case is for the number of warehouses
                // varying the number of warehouses also varies contention in the system.
                // if there are less warehouses, there are more shared updates on the same tables, leading to more contention
                // if there are more warehouses, there are less shared updates on the same tables, leading to less contention
            errno = 0;
            tpcc_config.num_warehouses = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of warehouses");
            }
            break;
        case 'a': // this case is for the skew factor
                // skew factor is a measure of how skewed the data is in the database
                // skew means that some records are accessed more frequently than others
                // a skew factor of 0 means that all records are accessed equally
                // a skew factor of 0.99 means that one record is accessed all the time
            errno = 0;
            ycsb_config.skew_factor = strtod(optarg, &end_char);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid skew factor");
            }
            break;
        case 'r': // this case is for the full record read
                // whether or not to read the full record, regardless of field splitting, i.e. whether to read all fields of a record
            if (std::string(optarg) == "true")
            {
                ycsb_config.full_record_read = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.full_record_read = false;
            }
            else
            {
                throw std::runtime_error("Invalid full record read");
            }
            break;
        case 'c': // this case is for the number of CPU execution threads
                // the number of CPU execution threads is the number of threads that run on the CPU
                // the more threads there are, the more parallelism there is in the system
                // the less threads there are, the less parallelism there is in the system
            errno = 0;
            ycsb_config.cpu_exec_num_threads = strtoul(optarg, &end_char, 0);
            tpcc_config.cpu_exec_num_threads = ycsb_config.cpu_exec_num_threads;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of CPU execution threads");
            }
            break;
        case 'e': // this case is for the number of epochs
                 // an epoch is a period of time in which all transactions are executed
                // the more epochs there are, the more transactions are executed
               // the less epochs there are, the less transactions are executed
            errno = 0;
            tpcc_config.epochs = strtoul(optarg, &end_char, 0);
            ycsb_config.epochs = tpcc_config.epochs;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of epochs");
            }
            break;
        case 's': // this case is for the number of transactions
                // the number of transactions is the number of transactions that are executed per epoch
            errno = 0;
            tpcc_config.num_txns = strtoul(optarg, &end_char, 0);
            ycsb_config.num_txns = tpcc_config.num_txns;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of transactions");
            }
            break;
        case 'f': // this case is for the split fields
                // whether or not to split fields, i.e. whether to split the fields of a record
            if (std::string(optarg) == "true")
            {
                ycsb_config.split_field = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.split_field = false;
            }
            else
            {
                throw std::runtime_error("Invalid split fields");
            }
            break;
        case 'm': // this case is for the commutative operations
                // whether or not to use commutative operations
                // this lets us know whether or not to use atomic operations rather than locks later on(line 262)
                // since the order of operations dont matter, we have no need for locks
                // This is specific to NewOrder transactions in Tpcc
            if (std::string(optarg) == "true")
            {
                commutative_ops = true;
            }
            else if (std::string(optarg) == "false")
            {
                commutative_ops = false;
            }
            else
            {
                throw std::runtime_error("Invalid commutative ops");
            }
            break;
        case 'n': // this case is for the number of records
            errno = 0;
            ycsb_config.num_records = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of records");
            }
            break;
        case 'x': // this case is for the execution device
                // it can chose to execute on the CPU or the GPU (the initialization still occurs on the GPU)
            if (std::string(optarg) == "cpu")
            {
                tpcc_config.execution_device = epic::DeviceType::CPU;
                ycsb_config.execution_device = epic::DeviceType::CPU;
            }
            else if (std::string(optarg) == "gpu")
            {
                tpcc_config.execution_device = epic::DeviceType::GPU;
                ycsb_config.execution_device = epic::DeviceType::GPU;
            }
            else
            {
                throw std::runtime_error("Invalid execution device");
            }
            break;
        default:
            throw std::runtime_error("Invalid option");
        }
    }

    /* this is a hack to run gacco NewOrder without holding locks on warehouse... */
    if (commutative_ops)
    {
        tpcc_config.gacco_use_atomic = true;
        tpcc_config.gacco_tpcc_stock_use_atomic = true;
    }
    else if (tpcc_config.txn_mix.new_order > 0)
    {
        tpcc_config.gacco_use_atomic = true;
        tpcc_config.gacco_tpcc_stock_use_atomic = false;
    } else {
        tpcc_config.gacco_use_atomic = false;
        tpcc_config.gacco_tpcc_stock_use_atomic = false;
    }

    std::unique_ptr<epic::Benchmark> benchmark;
    if (bench == "tpcc")
    {
        if (db == "epic")
        {
            benchmark = std::make_unique<epic::tpcc::TpccDb>(tpcc_config);
        }
        else if (db == "gacco")
        {
            benchmark = std::make_unique<gacco::tpcc::TpccDb>(tpcc_config);
        }
        else
        {
            throw std::runtime_error("Invalid database name");
        }
    }
    else if (bench == "ycsb")
    {
        if (db == "epic")
        {
            benchmark = std::make_unique<epic::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else if (db == "gacco")
        {
            benchmark = std::make_unique<gacco::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else
        {
            throw std::runtime_error("Invalid database name");
        }
    }
    benchmark->loadInitialData(); // traces back to ycsb_gpu_index.cu line 140
    benchmark->generateTxns(); // traces back to tpcc.cpp
    benchmark->runBenchmark();

    return 0;
}