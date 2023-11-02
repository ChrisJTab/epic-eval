//
// Created by Shujian Qian on 2023-08-23.
//

#ifndef TPCC_TXN_H
#define TPCC_TXN_H

#include <cstdlib>

#include "txn.h"

namespace epic::tpcc {

enum class TpccTxnType : uint32_t
{
    NEW_ORDER = 1,
    PAYMENT,
    ORDER_STATUS,
    DELIVERY,
    STOCK_LEVEL
};

struct FixedSizeTxn
{
    static constexpr size_t kMaxItems = 15;
};

struct VariableSizeTxn
{
    static constexpr size_t kMaxItems = 0;
};

/**
 * This is generated during the batching process and used by the indexer.
 *
 * @tparam TxnType
 */
template<typename TxnType>
struct NewOrderTxnInput
{
    struct ItemIndex
    {
        uint32_t i_id;
        uint32_t w_id;
        uint32_t order_quantities;
    };
    uint32_t origin_w_id;
    uint32_t d_id;
    uint32_t c_id;
    uint32_t o_id;
    uint32_t num_items;
    ItemIndex items[TxnType::kMaxItems];
};

/**
 * This is generated by the indexer through indexing the batched transaction.
 * It is used by the execution planner as well as the execution engine (along with the execution plan).
 *
 * @tparam TxnType
 */
template<typename TxnType>
struct NewOrderTxnParams
{
    struct ItemParams
    {
        uint32_t item_id;  /* i_id */
        uint32_t stock_id; /* w_id, i_id */
        uint32_t order_line_id;
        uint32_t order_quantities;
    };
    uint32_t warehouse_id; /* w_id */
    uint32_t district_id;  /* w_id, d_id */
    uint32_t customer_id;  /* w_id, d_id, c_id */
    uint32_t new_order_id; /* w_id, d_id, o_id */
    uint32_t order_id;     /* w_id, d_id, o_id */
    uint32_t num_items;
    bool all_local;
    ItemParams items[TxnType::kMaxItems];
};

/**
 * This is generated by the execution planner and used by the execution engine.
 *
 * @tparam TxnType
 */
template<typename TxnType>
struct NewOrderExecPlan
{
    struct ItemPlan
    {
        uint32_t item_loc;
        uint32_t stock_read_loc;
        uint32_t stock_write_loc;
        uint32_t orderline_loc;
    };
    uint32_t warehouse_loc;
    uint32_t district_loc;
    uint32_t customer_loc;
    uint32_t new_order_loc;
    uint32_t order_loc;
    ItemPlan item_plans[TxnType::kMaxItems];
};

struct PaymentTxnInput
{
    uint32_t warehouse_id;
    uint32_t district_id;
    uint32_t customer_warehouse_id;
    uint32_t customer_district_id;
    uint32_t payment_amount;
    uint32_t customer_id;
};

struct PaymentTxnParams
{
    uint32_t warehouse_id;
    uint32_t district_id;
    uint32_t customer_id;
    uint32_t payment_amount;
};

struct PaymentTxnExecPlan
{
    uint32_t warehouse_read_loc;
    uint32_t warehouse_write_loc;
    uint32_t district_read_loc;
    uint32_t district_write_loc;
    uint32_t customer_read_loc;
    uint32_t customer_write_loc;
};

class OrderStatusTxn
{
    /* TODO: implement order-status txn */
};

class DeliveryTxn
{
    /* TODO: implement delivery txn */
};

class StockLevelTxn
{
    /* TODO: implement stock-level txn */
};

/**
 * Make is easier to calculate the size of the largest txn.
 */
union TpccTxn
{
    NewOrderTxnInput<FixedSizeTxn> new_order_txn;
    PaymentTxnInput payment_txn;
    OrderStatusTxn order_status_txn;
    DeliveryTxn delivery_txn;
    StockLevelTxn stock_level_txn;
};

union TpccTxnParam
{
    NewOrderTxnParams<FixedSizeTxn> new_order_txn;
    PaymentTxnParams payment_txn;
} __attribute__((aligned(4)));

union TpccExecPlan
{
    NewOrderExecPlan<FixedSizeTxn> new_order_txn;
    PaymentTxnExecPlan payment_txn;
} __attribute__((aligned(4)));

/* TODO: how to implement piece exection on GPU? */
void runTransaction(BaseTxn *txn);
void runTransaction(NewOrderTxnInput<FixedSizeTxn> *txn);
void runTransaction(PaymentTxnInput *txn);
void runTransaction(OrderStatusTxn *txn);
void runTransaction(DeliveryTxn *txn);
} // namespace epic::tpcc

#endif // TPCC_TXN_H
