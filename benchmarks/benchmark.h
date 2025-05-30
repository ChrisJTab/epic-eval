//
// Created by Shujian Qian on 2023-11-22.
//

// this checks whether the header is already defined, if not, it defines it
#ifndef EPIC_BENCHMARKS_BENCHMARK_H
#define EPIC_BENCHMARKS_BENCHMARK_H

namespace epic {

class Benchmark
{
public:
    virtual ~Benchmark() = default;

    virtual void loadInitialData() = 0;
    virtual void generateTxns() = 0;
    virtual void runBenchmark() = 0;
};

}

#endif // EPIC_BENCHMARKS_BENCHMARK_H
