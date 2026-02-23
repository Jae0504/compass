#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <exception>
#include <functional>

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.emplace_back([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);
                    if (id >= end) break;

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}
