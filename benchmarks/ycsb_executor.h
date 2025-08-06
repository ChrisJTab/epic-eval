//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_EXECUTOR_H
#define EPIC_BENCHMARKS_YCSB_EXECUTOR_H

#include <txn.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_storage.h>
#include <benchmarks/ycsb_config.h>

namespace epic::ycsb {

class Executor
{
public:
    std::variant<YcsbFullLayout, YcsbSplitLayout> layout;
    TxnArray<YcsbTxnParam> txn;
    TxnArray<YcsbExecPlan> plan;
    YcsbConfig config;
    Executor(std::variant<YcsbFullLayout, YcsbSplitLayout> lay, TxnArray<YcsbTxnParam> txn,
        TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : layout(std::move(lay))
        , txn(txn)
        , plan(plan)
        , config(config)
    {}
    virtual ~Executor() = default;
    virtual void execute(uint32_t epoch)
    {
        throw std::runtime_error("epic::ycsb::Executor::execute() is not implemented.");
    };
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_EXECUTOR_H
