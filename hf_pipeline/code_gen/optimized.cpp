#include <iostream>
#include <vector>
#include <ctime>
#include <limits>

// Linear Congruential Generator (LCG)
class LCG {
public:
    using ull = unsigned long long;
    LCG(ull seed, ull a = 1664525, ull c = 1013904223, ull m = 1ull << 32)
        : value(seed), a(a), c(c), m(m) {}

    ull next() {
        value = (a * value + c) % m;
        return value;
    }

private:
    ull value, a, c, m;
};

// Find maximum subarray sum using Kadane's Algorithm
long long max_subarray_sum(const std::vector<int>& random_numbers) {
    long long max_sum = LLONG_MIN;
    long long current_sum = 0;
    for (int num : random_numbers) {
        current_sum += num;
        if (current_sum > max_sum) max_sum = current_sum;
        if (current_sum < 0) current_sum = 0;
    }
    return max_sum;
}

long long total_max_subarray_sum(int n, unsigned int initial_seed, int min_val, int max_val) {
    long long total_sum = 0;
    LCG lcg_gen(initial_seed);
    
    for (int run = 0; run < 20; ++run) {
        unsigned int seed = lcg_gen.next();
        LCG run_lcg(seed);
        std::vector<int> random_numbers(n);
        for (int i = 0; i < n; ++i) {
            random_numbers[i] = run_lcg.next() % (max_val - min_val + 1) + min_val;
        }
        total_sum += max_subarray_sum(random_numbers);
    }
    return total_sum;
}

int main() {
    int n = 1000;
    unsigned int initial_seed = 42;
    int min_val = -10;
    int max_val = 10;

    clock_t start_time = clock();
    long long result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    clock_t end_time = clock();

    std::cout << "Total Maximum Subarray Sum (20 runs): " << result << std::endl;
    std::cout << "Execution Time: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << std::endl;

    return 0;
}