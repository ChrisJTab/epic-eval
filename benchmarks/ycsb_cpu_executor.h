//
// Created by Shujian Qian on 2023-12-05.
//

#ifndef EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H
#define EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H

#include <benchmarks/ycsb_executor.h>

namespace epic::ycsb {

class CpuExecutor : public Executor
{
public:
    // Note that there is no split layout
    using FullLayout = YcsbFullLayout;   // for full-field YCSB

    CpuExecutor(FullLayout layout, TxnArray<YcsbTxnParam> txn,
        TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : Executor(std::move(layout), txn, plan, config){};
    ~CpuExecutor() override = default;

    void execute(uint32_t epoch) override;
private:
    void executionWorker(uint32_t epoch, uint32_t thread_id);
};

}

#endif // EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H
