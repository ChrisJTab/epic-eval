//
// Created by Shujian Qian on 2023-09-15.
//
/*
Index: is a shared pointer who's type is chosen at runtime
If the device is the CPU, then index = std::make_shared<TpccCpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
This is a CPU hash table index

If the device is the GPU, then index = std::make_shared<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
This is a GPU hash table index

The index represents the primary key to recordID hash table that every transaction consults before it runs

index->loadInitialData() — at start-up it bulk-loads the table keys so that the hash table already contains every pre-existing record.
This is done in tpcc_index.cpp, where all data is loaded



*/

#include "benchmarks/tpcc.h"

#include <cassert>
#include <random>
#include <cstdio>
#include <chrono>

#include "benchmarks/tpcc_txn.h"

#include "util_log.h"
#include "util_device_type.h"
#include "gpu_txn.h"
#include "gpu_allocator.h"
#include "gpu_execution_planner.h"
#include "benchmarks/tpcc_gpu_submitter.h"
#include "benchmarks/tpcc_executor.h"
#include "benchmarks/tpcc_gpu_executor.h"
#include <benchmarks/tpcc_txn_gen.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_cpu_executor.h>

namespace epic::tpcc {
TpccTxnMix::TpccTxnMix(
    uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level)
    : new_order(new_order)
    , payment(payment)
    , order_status(order_status)
    , delivery(delivery)
    , stock_level(stock_level)
{
    assert(new_order + payment + order_status + delivery + stock_level == 100);
}

TpccDb::TpccDb(TpccConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , execution_param_input(config.num_txns, config.execution_device, false)
    , execution_plan_input(config.num_txns, config.execution_device, false)
    , cpu_aux_index(config)
    , gpu_aux_index(config)
    , packed_txn_array_builder(config.num_txns)
{
    //    index = std::make_shared<TpccCpuIndex>(config);
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TpccTxnArrayT(config.num_txns, DeviceType::CPU);
    }
    if (config.index_device == DeviceType::CPU)
    {
        index = std::make_shared<TpccCpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
    }
    else if (config.index_device == DeviceType::GPU)
    {
        index = std::make_shared<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
    }
    else
    {
        throw std::runtime_error("Unsupported index device");
    }

    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    index_execution_param_bridge.Link(index_output, execution_param_input);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);

    if (config.initialize_device == DeviceType::GPU)
    {
        GpuAllocator allocator;
        warehouse_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "warehouse", allocator, 0, 2, config.num_txns, config.num_warehouses, initialization_output);
        district_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "district", allocator, 0, 2, config.num_txns, config.num_warehouses * 10, initialization_output);
        customer_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "customer", allocator, 0, 20, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        history_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "history", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        new_order_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "new_order", allocator, 0, 10, config.num_txns, config.num_warehouses * 10 * 900, initialization_output);
        order_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "order", allocator, 0, 20, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        order_line_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>("order_line", allocator,
            0, 30, config.num_txns, config.num_warehouses * 10 * 3000 * 15, initialization_output);
        item_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "item", allocator, 0, 15, config.num_txns, 100'000, initialization_output);
        stock_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
            "stock", allocator, 0, 15 * 2, config.num_txns, 100'000 * config.num_warehouses, initialization_output);

        warehouse_planner->Initialize();
        district_planner->Initialize();
        customer_planner->Initialize();
        history_planner->Initialize();
        new_order_planner->Initialize();
        order_planner->Initialize();
        order_line_planner->Initialize();
        item_planner->Initialize();
        stock_planner->Initialize();
        allocator.PrintMemoryInfo();

        using SubmitDestT = TpccGpuSubmitter<TpccTxnParamArrayT>::TableSubmitDest;
        submitter = std::make_shared<TpccGpuSubmitter<TpccTxnParamArrayT>>(
            SubmitDestT{warehouse_planner->d_num_ops, warehouse_planner->d_op_offsets,
                warehouse_planner->d_submitted_ops, warehouse_planner->d_scratch_array,
                warehouse_planner->scratch_array_bytes, warehouse_planner->curr_num_ops},
            SubmitDestT{district_planner->d_num_ops, district_planner->d_op_offsets,
                district_planner->d_submitted_ops, district_planner->d_scratch_array,
                district_planner->scratch_array_bytes, district_planner->curr_num_ops},
            SubmitDestT{customer_planner->d_num_ops, customer_planner->d_op_offsets,
                customer_planner->d_submitted_ops, customer_planner->d_scratch_array,
                customer_planner->scratch_array_bytes, customer_planner->curr_num_ops},
            SubmitDestT{history_planner->d_num_ops, history_planner->d_op_offsets,
                history_planner->d_submitted_ops, history_planner->d_scratch_array,
                history_planner->scratch_array_bytes, history_planner->curr_num_ops},
            SubmitDestT{new_order_planner->d_num_ops, new_order_planner->d_op_offsets,
                new_order_planner->d_submitted_ops, new_order_planner->d_scratch_array,
                new_order_planner->scratch_array_bytes, new_order_planner->curr_num_ops},
            SubmitDestT{order_planner->d_num_ops, order_planner->d_op_offsets,
                order_planner->d_submitted_ops, order_planner->d_scratch_array, order_planner->scratch_array_bytes,
                order_planner->curr_num_ops},
            SubmitDestT{order_line_planner->d_num_ops, order_line_planner->d_op_offsets,
                order_line_planner->d_submitted_ops, order_line_planner->d_scratch_array,
                order_line_planner->scratch_array_bytes, order_line_planner->curr_num_ops},
            SubmitDestT{item_planner->d_num_ops, item_planner->d_op_offsets,
                item_planner->d_submitted_ops, item_planner->d_scratch_array, item_planner->scratch_array_bytes,
                item_planner->curr_num_ops},
            SubmitDestT{stock_planner->d_num_ops, stock_planner->d_op_offsets,
                stock_planner->d_submitted_ops, stock_planner->d_scratch_array, stock_planner->scratch_array_bytes,
                stock_planner->curr_num_ops});
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }

    if (config.execution_device == DeviceType::GPU)
    {
        /* TODO: initialize records & versions */
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        GpuAllocator allocator;

        /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
        size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.warehouseTableSize();
        size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
        logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
            formatSizeBytes(warehouse_ver_size));
        records.warehouse_record = static_cast<Record<WarehouseValue> *>(allocator.Allocate(warehouse_rec_size));
        versions.warehouse_version = static_cast<Version<WarehouseValue> *>(allocator.Allocate(warehouse_ver_size));

        size_t district_rec_size = sizeof(Record<DistrictValue>) * config.districtTableSize();
        size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
        logger.Info(
            "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
        records.district_record = static_cast<Record<DistrictValue> *>(allocator.Allocate(district_rec_size));
        versions.district_version = static_cast<Version<DistrictValue> *>(allocator.Allocate(district_ver_size));

        size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.customerTableSize();
        size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
        logger.Info(
            "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
        records.customer_record = static_cast<Record<CustomerValue> *>(allocator.Allocate(customer_rec_size));
        versions.customer_version = static_cast<Version<CustomerValue> *>(allocator.Allocate(customer_ver_size));

        /* TODO: history table is too big */
        //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
        //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
        //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
        //                    formatSizeBytes(history_ver_size));
        //        records.history_record = static_cast<Record<HistoryValue> *>(allocator.Allocate(history_rec_size));
        //        versions.history_version = static_cast<Version<HistoryValue> *>(allocator.Allocate(history_ver_size));

        size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.newOrderTableSize();
        size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
            formatSizeBytes(new_order_ver_size));
        records.new_order_record = static_cast<Record<NewOrderValue> *>(allocator.Allocate(new_order_rec_size));
        versions.new_order_version = static_cast<Version<NewOrderValue> *>(allocator.Allocate(new_order_ver_size));

        size_t order_rec_size = sizeof(Record<OrderValue>) * config.orderTableSize();
        size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
        records.order_record = static_cast<Record<OrderValue> *>(allocator.Allocate(order_rec_size));
        versions.order_version = static_cast<Version<OrderValue> *>(allocator.Allocate(order_ver_size));

        size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.orderLineTableSize();
        size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
            formatSizeBytes(order_line_ver_size));
        records.order_line_record = static_cast<Record<OrderLineValue> *>(allocator.Allocate(order_line_rec_size));
        versions.order_line_version = static_cast<Version<OrderLineValue> *>(allocator.Allocate(order_line_ver_size));

        size_t item_rec_size = sizeof(Record<ItemValue>) * config.itemTableSize();
        size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
        records.item_record = static_cast<Record<ItemValue> *>(allocator.Allocate(item_rec_size));
        versions.item_version = static_cast<Version<ItemValue> *>(allocator.Allocate(item_ver_size));

        size_t stock_rec_size = sizeof(Record<StockValue>) * config.stockTableSize();
        size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
        logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
        records.stock_record = static_cast<Record<StockValue> *>(allocator.Allocate(stock_rec_size));
        versions.stock_version = static_cast<Version<StockValue> *>(allocator.Allocate(stock_ver_size));

        allocator.PrintMemoryInfo();

        /* TODO: execution input need to be transferred too, currently using placeholders */
//        executor =
//            std::make_shared<GpuExecutor>(records, versions, initialization_input, initialization_output, config);
        executor = std::make_shared<GpuExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>>(
            records, versions, execution_param_input, execution_plan_input, config);
    }
    else if (config.execution_device == DeviceType::CPU)
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
        size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.warehouseTableSize();
        size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
        logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
                    formatSizeBytes(warehouse_ver_size));
        records.warehouse_record = static_cast<Record<WarehouseValue> *>(Malloc(warehouse_rec_size));
        versions.warehouse_version = static_cast<Version<WarehouseValue> *>(Malloc(warehouse_ver_size));

        size_t district_rec_size = sizeof(Record<DistrictValue>) * config.districtTableSize();
        size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
        logger.Info(
            "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
        records.district_record = static_cast<Record<DistrictValue> *>(Malloc(district_rec_size));
        versions.district_version = static_cast<Version<DistrictValue> *>(Malloc(district_ver_size));

        size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.customerTableSize();
        size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
        logger.Info(
            "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
        records.customer_record = static_cast<Record<CustomerValue> *>(Malloc(customer_rec_size));
        versions.customer_version = static_cast<Version<CustomerValue> *>(Malloc(customer_ver_size));

        /* TODO: history table is too big */
        //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
        //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
        //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
        //                    formatSizeBytes(history_ver_size));
        //        records.history_record = static_cast<Record<HistoryValue> *>(Malloc(history_rec_size));
        //        versions.history_version = static_cast<Version<HistoryValue> *>(Malloc(history_ver_size));

        size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.newOrderTableSize();
        size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
                    formatSizeBytes(new_order_ver_size));
        records.new_order_record = static_cast<Record<NewOrderValue> *>(Malloc(new_order_rec_size));
        versions.new_order_version = static_cast<Version<NewOrderValue> *>(Malloc(new_order_ver_size));

        size_t order_rec_size = sizeof(Record<OrderValue>) * config.orderTableSize();
        size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
        records.order_record = static_cast<Record<OrderValue> *>(Malloc(order_rec_size));
        versions.order_version = static_cast<Version<OrderValue> *>(Malloc(order_ver_size));

        size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.orderLineTableSize();
        size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
                    formatSizeBytes(order_line_ver_size));
        records.order_line_record = static_cast<Record<OrderLineValue> *>(Malloc(order_line_rec_size));
        versions.order_line_version = static_cast<Version<OrderLineValue> *>(Malloc(order_line_ver_size));

        size_t item_rec_size = sizeof(Record<ItemValue>) * config.itemTableSize();
        size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
        records.item_record = static_cast<Record<ItemValue> *>(Malloc(item_rec_size));
        versions.item_version = static_cast<Version<ItemValue> *>(Malloc(item_ver_size));

        size_t stock_rec_size = sizeof(Record<StockValue>) * config.stockTableSize();
        size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
        logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
        records.stock_record = static_cast<Record<StockValue> *>(Malloc(stock_rec_size));
        versions.stock_version = static_cast<Version<StockValue> *>(Malloc(stock_ver_size));
        executor = std::make_shared<CpuExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>>(
            records, versions, execution_param_input, execution_plan_input, config);
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }
}

/*
 * generateTxns()
 * --------------
 * Pre-build the entire TPC-C workload in host RAM.
 *
 * For each epoch:
 *   • Draw NUM_TXNS transaction types from the configured mix
 *     (TpccTxnGenerator::getTxnType).
 *   • Serialize each transaction into the epoch’s TpccTxnArrayT
 *     as a variable-length blob:
 *         | BaseTxn header | concrete txn payload |
 *     The start offset of txn i is stored in index[i];
 *     ‘size’ tracks the running end-of-blob cursor.
 *   • No heap allocations occur after this; later stages can
 *     DMA the packed byte-array to the GPU in one shot.
 *
 * Result: txn_array[epoch] now contains a fully-specified,
 * reproducible batch ready for the indexing/initialization/
 * execution pipeline.
 * 
 * Note that the actual generation/creation of the transaction occurs in tpcc_txn_gen.cpp,
 * where the generator.generateTxn(txn_type, txn, timestamp); occurs.
 */
void TpccDb::generateTxns()
{
    auto &logger = Logger::GetInstance(); 

    TpccTxnGenerator generator(config); // ① RNG + Zipf helpers seeded
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) // ② Outer loop: batches
    {
        logger.Info("Generating epoch {}", epoch);
        TpccTxnArrayT &txn_input_array = txn_array[epoch];
        uint32_t curr_size = 0; // ③ Running write cursor
        for (size_t i = 0; i < config.num_txns; ++i)  // ④ Inner loop: one txn
        {
#if 0
            BaseTxn *txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn, timestamp);
#else
            TpccTxnType txn_type = generator.getTxnType();                                      // ⑤ Draw from mix %
            constexpr uint32_t txn_sizes[6] = {0, BaseTxnSize<NewOrderTxnInput<FixedSizeTxn>>::value,
                BaseTxnSize<PaymentTxnInput>::value, BaseTxnSize<OrderStatusTxnInput>::value,
                BaseTxnSize<DeliveryTxnInput>::value, BaseTxnSize<StockLevelTxnInput>::value};  // per-type byte count
            txn_input_array.index[i] = curr_size;                                               // ⑥ Offset table
            curr_size += txn_sizes[static_cast<uint32_t>(txn_type)];                            // ⑦ Advance cursor
            txn_input_array.size = curr_size;                                                   // ⑧ Total bytes so far
            BaseTxn *txn = txn_input_array.getTxn(i);                                           // ⑨ Pointer into blob
            uint32_t timestamp = epoch * config.num_txns + i;                                   // ⑩ Global timestamp
            generator.generateTxn(txn_type, txn, timestamp);                                    // ⑪ Populate fields
#endif
        }
    }
}

void TpccDb::loadInitialData()
{
    index->loadInitialData();
    /*
    The line above builds the primary key hash tables using static_map(GPU) or unordered_map(CPU) per base table
    These maps support exact-match lookups and inserts for every read or write in every transaction

    Bulk-inserts every existing row’s primary key into its table-specific hash map, assigning row-IDs 0…N-1. 
    When epoch 1 starts, the primary index can answer “does key X exist and where?” immediately.
    */
    // cpu_aux_index.loadInitialData(); /* cpu aux index replace by gpu aux index */
    gpu_aux_index.loadInitialData();

    /*
    A single GPU B-link-tree ("aux index") plus three side arrays
    - co_btree maps each table to a dummy value and is ordered the OrderID so it can answer the "latest order of a customer" query

    Basically, the primary index is for fast exact keys and rowID translation for all the tables
    The auxiliary index is a fast range/topK query on customer order relationsips and cached order facts

    Walks the order seed data (3 000 orders / district) and 
    inserts (w,d,c,o) keys into co_btree so that a descending scan over the key space returns the most recent order first.
    Fills the three side arrays with pre-generated num_items, customer_id, and 15 random item IDs per order. These arrays let kernels 
    answer “how many items were in order O?” without touching the tables.

    */
}


/*
 * runBenchmark()
 * --------------
 * Orchestrates the full per-epoch pipeline:
 *
 *   1.  (optional) legacy CPU aux-index timing
 *   2.  Copy packed txns of epoch e-1 to GPU and build GPU-friendly views
 *   3.  GPU auxiliary index:
 *         • insert NEW_ORDER keys into B-link tree
 *         • service ORDER_STATUS / DELIVERY / STOCK_LEVEL range queries
 *   4.  Primary hash indexing: logical PK → dense Row-ID
 *   5.  Ship indexed payloads to the initialization stage
 *   6.  Submit per-table op queues to planners
 *   7.  Planners map Row-IDs to concrete addresses (RECORD_A/B, scratch),
 *       producing a compact execution plan per txn
 *   8.  Transfer execution params + plans to executor buffers
 *   9.  Launch epoch execution kernels (reads, writes, commits)
 *
 * Every region is timed and logged so micro-benchmarks can report
 *   “indexing  = X µs,   init = Y µs,   execute = Z µs”.
 * The pipeline processes epoch e-1 while the host prepares epoch e,
 * maximising overlap between PCIe copies and GPU compute.
 */
void TpccDb::runBenchmark()
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (uint32_t epoch_id = 1; epoch_id <= config.epochs; ++epoch_id)            // Loop over all epochs
    {
        logger.Info("Running epoch {}", epoch_id);

        /* cpu aux index */
        {
            start_time = std::chrono::high_resolution_clock::now();

            uint32_t index_epoch_id = epoch_id - 1;
            /* cpu_aux_index replaced by gpu_aux_index */
            // cpu_aux_index.insertTxnUpdates(txn_array[index_epoch_id], epoch_id);
            // cpu_aux_index.performRangeQueries(txn_array[index_epoch_id], epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} cpu aux index time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

            /*
            The project migrated to a GPU-resident B-link tree (gpu_aux_index) which is orders of magnitude faster.

            The CPU version is left in the codebase only for regression experiments and A/B timing, 
            so the timing block remains but the function calls are disabled.
            */
        }

        /* transfer */
        /*
        Copy packed transactions of epoch e-1 to GPU and build GPU-friendly views
        */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1; // take the previous epoch's txns
            input_index_bridge.Link(txn_array[index_epoch_id], index_input); // Link where the source is the transaction array, and the destination is the index input
            input_index_bridge.StartTransfer(); // transfer transactions to the index input
            input_index_bridge.FinishTransfer();
            // Index output for the primary hash index to use. Essentially, index_output just holds the indexes of each transaction in the transaction array.
            packed_txn_array_builder.buildPackedTxnArrayGpu(index_input, index_output); 
            packed_txn_array_builder.buildPackedTxnArrayGpu(index_input, initialization_output); // fed to the execution-plan builders; they also prefer the packed layout

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxn>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, index_input.txns, copy_size);

                for (int i = 0; i < print_size; ++i)
                {
                    auto param = reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->txn_type;
                    logger.Info("txn {} type {}", i, param);
                }
                logger.flush();
            }

#endif
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }


        /* ──────────────────────── GPU auxiliary-index phase ───────────────────────── */
        /* The “aux index” is a *separate* GPU B-link-tree that supports the             */
        /* range-style lookups TPCC needs (                                                     */
        /*   – “latest order of customer C” (ORDER-STATUS)                                 */
        /*   – “oldest new-order per district”  (DELIVERY)                                 */
        /*   – “20 most recent orders”          (STOCK-LEVEL)                              */
        /* ) faster than doing a full hash-table probe + table scan.                        */
        
        /* gpu aux index */
        {

            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            /*  Apply INSERT-only mutations produced by epoch e-1                         *
            *     ------------------------------------------------------------------------ *
            *  Each NEW_ORDER txn creates:                                                 *
            *    key = (w,d,c,/**descending** o_id)  value = dummy (0)                    *
            *  The kernel:                                                                 *
            *    • bulk-inserts those keys into the GPU B-link-tree (`co_btree`)           *
            *    • caches ‘num_items’, ‘customer_id’ and 15 item IDs into three flat       *
            *      side arrays for later use by range queries.                             */
            gpu_aux_index.insertTxnUpdates(index_input, index_epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} gpu aux index time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

            start_time = std::chrono::high_resolution_clock::now();
            /*  Satisfy RANGE queries required for epoch e-1 txns                         *
            *     ------------------------------------------------------------------------ *
            *  Using the freshly updated B-tree + caches, a second kernel walks every      *
            *  ORDER-STATUS / DELIVERY / STOCK-LEVEL txn in **index_input** and:           *
            *      • looks up / scans the tree                                             *
            *      • writes the answers (latest O_ID, customer id, unique item IDs …)      *
            *        directly into the corresponding *parameter block* inside              *
            *        **index_output**                                                     *
            *  Doing this now means the later hash-index pass and execution planner        *
            *  already see the range-query results without extra work.                     */
            gpu_aux_index.performRangeQueries(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} gpu aux index part2 time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            /*
            Updates the primary hash tables with new rows generated in the current epoch
            Translates every logical key inside the transactions of epoch e-1 into concrete rowIDs, storing them into index_output
            */
            index->indexTxns(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            /* `index_initialization_bridge` was linked earlier like so:
                index_initialization_bridge.Link(index_output, initialization_input);

            • **index_output**         – PackedTxnArray<TpccTxnParam> that now contains *fully-bound* param blocks
                                            (written a few lines above by `index->indexTxns()`).
                                            It lives on the **index device** (CPU if you chose the CPU indexer,
                                            GPU if you chose the GPU indexer).

            • **initialization_input** – Another PackedTxnArray<TpccTxnParam> located on the
                                            **initialisation device** (always the GPU in the current build).
                                            Execution-planner kernels will read from this buffer next.

            The bridge abstracts away the nasty `cudaMemcpyAsync` vs. `memcpy` dance.          */
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} init_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, initialization_input.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto param = &reinterpret_cast<TpccTxnParam *>(
                        reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->data)
                        ->new_order_txn;
                    logger.Info("txn {} warehouse[{}] district[{}] customer[{}] order[{}] new_order[{}] numitems[{}] "
                                "item1[{}] stock_read1[{}] order_line1[{}] quantity1[{}] "
                                "item2[{}] stock_read2[{}] order_line2[{}] quantity2[{}] "
                                "item3[{}] stock_read3[{}] order_line3[{}] quantity3[{}] "
                                "item4[{}] stock_read4[{}] order_line4[{}] quantity4[{}] "
                                "item5[{}] stock_read5[{}] order_line5[{}] quantity5[{}] ",
                                i, param->warehouse_id, param->district_id, param->customer_id, param->order_id,
                                param->new_order_id, param->num_items, param->items[0].item_id, param->items[0].stock_id,
                                param->items[0].order_line_id, param->items[0].order_quantities, param->items[1].item_id,
                                param->items[1].stock_id, param->items[1].order_line_id, param->items[1].order_quantities,
                                param->items[2].item_id, param->items[2].stock_id, param->items[2].order_line_id,
                                param->items[2].order_quantities, param->items[3].item_id, param->items[3].stock_id,
                                param->items[3].order_line_id, param->items[3].order_quantities, param->items[4].item_id,
                                param->items[4].stock_id, param->items[4].order_line_id, param->items[4].order_quantities);
                }
                logger.flush();
            }
#endif
        }

        /* submit */
        /* ────────────────────── “submission” step ─────────────────────────── */
        /* Tell every per-table **execution-planner** how many row‐operations     *
        * the coming epoch will issue, and copy that metadata into their         *
        * GPU scratch buffers.                                                   *
        *                                                                        *
        * After this step the planners can allocate/resize internal arrays       *
        * and build record-location maps without re-scanning the whole workload. */
        {
            /* 
            `initialization_input` is a PackedTxnArray<TpccTxnParam>, already on the GPU.
            It contains, for every transaction of epoch e, the fully-bound *parameter block*
            that tells the execution-planner kernels which row-IDs the txn will touch.
            
            `submitter` is a <TpccSubmitter<TpccTxnParamArrayT>> created in the
            constructor with one “TableSubmitDest” per TPCC table:

                    SubmitDest {
                        d_num_ops        ←  global counter   (GPU)
                        d_op_offsets     ←  prefix-sum array (GPU)
                        d_submitted_ops  ←  op list buffer   (GPU)
                        d_scratch_array  ←  temporary byte-arena
                        scratch_bytes    ←  its size
                        curr_num_ops     ←  host mirror of d_num_ops
                    }

            The submitter walks **initialization_input** (the packed array of
            `TpccTxnParam` we just copied to the GPU) and, for every *logical*
            read / write in every transaction:

                • increments the matching table’s `d_num_ops`
                • appends a compact <rowID, op-type> entry into `d_submitted_ops`

            All updates are done with atomic adds inside a CUDA kernel so the
            whole pass is a single GPU launch. */
            start_time = std::chrono::high_resolution_clock::now();
            // submit pretty much stores all of the transactions in an array so we can do all the operations on them later in the execution planner
            submitter->submit(initialization_input);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} submission time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* initialize */
        {
            start_time = std::chrono::high_resolution_clock::now();

            /*
            This is where the execution planners take over, like where in the paper it did all of the scans, sorting, etc.
            */

            warehouse_planner->InitializeExecutionPlan();
            district_planner->InitializeExecutionPlan();
            customer_planner->InitializeExecutionPlan();
            history_planner->InitializeExecutionPlan();
            new_order_planner->InitializeExecutionPlan();
            order_planner->InitializeExecutionPlan();
            order_line_planner->InitializeExecutionPlan();
            item_planner->InitializeExecutionPlan();
            stock_planner->InitializeExecutionPlan();

            warehouse_planner->FinishInitialization();
            district_planner->FinishInitialization();
            customer_planner->FinishInitialization();
            history_planner->FinishInitialization();
            new_order_planner->FinishInitialization();
            order_planner->FinishInitialization();
            order_line_planner->FinishInitialization();
            item_planner->FinishInitialization();
            stock_planner->FinishInitialization();

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} initialization time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccExecPlan>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t execution_plan[max_print_size * base_txn_size];

                auto locToStr = [](uint32_t loc) -> std::string {
                    if (loc == loc_record_a)
                    {
                        return "RECORD_A";
                    }
                    else if (loc == loc_record_b)
                    {
                        return "RECORD_B";
                    }
                    else
                    {
                        return std::to_string(loc);
                    }
                };

                transferGpuToCpu(execution_plan, initialization_output.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto base_txn = reinterpret_cast<BaseTxn *>(execution_plan + i * base_txn_size);
                    if (1)
                    {
                        auto txn = reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(base_txn->data);
                        logger.Info("txn {} warehouse[{}] district[{}] customer[{}] new_order[{}] order[{}] "
                                    "item1[{}] stock_read1[{}] stock_write1[{}] order_line1[{}] "
                                    "item2[{}] stock_read2[{}] stock_write2[{}] order_line2[{}] "
                                    "item3[{}] stock_read3[{}] stock_write3[{}] order_line3[{}] "
                                    "item4[{}] stock_read4[{}] stock_write4[{}] order_line4[{}] "
                                    "item5[{}] stock_read5[{}] stock_write5[{}] order_line5[{}] ",
                            i, locToStr(txn->warehouse_loc), locToStr(txn->district_loc), locToStr(txn->customer_loc),
                            locToStr(txn->new_order_loc), locToStr(txn->order_loc),
                            locToStr(txn->item_plans[0].item_loc), locToStr(txn->item_plans[0].stock_read_loc),
                            locToStr(txn->item_plans[0].stock_write_loc), locToStr(txn->item_plans[0].orderline_loc),
                            locToStr(txn->item_plans[1].item_loc), locToStr(txn->item_plans[1].stock_read_loc),
                            locToStr(txn->item_plans[1].stock_write_loc), locToStr(txn->item_plans[1].orderline_loc),
                            locToStr(txn->item_plans[2].item_loc), locToStr(txn->item_plans[2].stock_read_loc),
                            locToStr(txn->item_plans[2].stock_write_loc), locToStr(txn->item_plans[2].orderline_loc),
                            locToStr(txn->item_plans[3].item_loc), locToStr(txn->item_plans[3].stock_read_loc),
                            locToStr(txn->item_plans[3].stock_write_loc), locToStr(txn->item_plans[3].orderline_loc),
                            locToStr(txn->item_plans[4].item_loc), locToStr(txn->item_plans[4].stock_read_loc),
                            locToStr(txn->item_plans[4].stock_write_loc), locToStr(txn->item_plans[4].orderline_loc));
                    }
                    if (0)
                    {
                        auto txn = reinterpret_cast<PaymentTxnExecPlan *>(base_txn->data);
                        logger.Info("txn {} warehouse[{}][{}] district[{}][{}] customer[{}][{}] history ", i,
                            locToStr(txn->warehouse_read_loc), locToStr(txn->warehouse_write_loc),
                            locToStr(txn->district_read_loc), locToStr(txn->district_write_loc),
                            locToStr(txn->customer_read_loc), locToStr(txn->customer_write_loc));
                    }
                }
            }
#endif
        }

        /* transfer */
        /*
        This step is a transfer to get the initialization execution plan to the execution input(the execution phase)
        */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_execution_param_bridge.StartTransfer();
            initialization_execution_plan_bridge.StartTransfer();
            index_execution_param_bridge.FinishTransfer();
            initialization_execution_plan_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* execution */
        {
#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, initialization_input.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto param = &reinterpret_cast<TpccTxnParam *>(
                        reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->data)
                                      ->new_order_txn;
                    logger.Info("txn {} warehouse[{}] district[{}] customer[{}] order[{}] new_order[{}] numitems[{}] "
                                "item1[{}] stock_read1[{}] order_line1[{}] quantity1[{}] "
                                "item2[{}] stock_read2[{}] order_line2[{}] quantity2[{}] "
                                "item3[{}] stock_read3[{}] order_line3[{}] quantity3[{}] "
                                "item4[{}] stock_read4[{}] order_line4[{}] quantity4[{}] "
                                "item5[{}] stock_read5[{}] order_line5[{}] quantity5[{}] ",
                        i, param->warehouse_id, param->district_id, param->customer_id, param->order_id,
                        param->new_order_id, param->num_items, param->items[0].item_id, param->items[0].stock_id,
                        param->items[0].order_line_id, param->items[0].order_quantities, param->items[1].item_id,
                        param->items[1].stock_id, param->items[1].order_line_id, param->items[1].order_quantities,
                        param->items[2].item_id, param->items[2].stock_id, param->items[2].order_line_id,
                        param->items[2].order_quantities, param->items[3].item_id, param->items[3].stock_id,
                        param->items[3].order_line_id, param->items[3].order_quantities, param->items[4].item_id,
                        param->items[4].stock_id, param->items[4].order_line_id, param->items[4].order_quantities);
                }
                logger.flush();
            }
#endif
            start_time = std::chrono::high_resolution_clock::now();
            /*
            actually executes... the executor can be either GpuExecutor or CpuExecutor, depending on the config
            */
            executor->execute(epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}
void TpccDb::indexEpoch(uint32_t epoch_id)
{
    /* TODO: remove */
    auto &logger = Logger::GetInstance();
    logger.Error("Deprecated function");
    exit(-1);

    //    /* zero-indexed */
    //    uint32_t index_epoch_id = epoch_id - 1;
    //
    //    /* it's important to index writes before reads */
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnWrites(txn, txn_param, index_epoch_id);
    //    }
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnReads(txn, txn_param, index_epoch_id);
    //    }
}
} // namespace epic::tpcc
