#include <iostream>
#include <vector>
#include <climits>
#include <chrono>

using namespace std;

// Linear Congruential Generator function
uint32_t lcg(uint32_t seed, int n, uint32_t a = 1664525, uint32_t c = 1013904223, uint32_t m = 4294967296ULL) {
    uint32_t value = seed;
    while (n--) {
        value = (a * value + c) % m;
    }
    return value;
}

// Function to calculate maximum subarray sum using Kadane's algorithm
int64_t max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    vector<int> random_numbers(n);
    uint32_t mod_range = max_val - min_val + 1;
    for (int i = 0; i < n; ++i) {
        seed = lcg(seed, 1);
        random_numbers[i] = seed % mod_range + min_val;
    }

    int64_t max_sum = LLONG_MIN;
    int64_t current_sum = 0;

    for (int j = 0; j < n; ++j) {
        current_sum = max(static_cast<int64_t>(random_numbers[j]), current_sum + random_numbers[j]);
        max_sum = max(max_sum, current_sum);
    }
    
    return max_sum;
}

// Function to calculate total maximum subarray sum across 20 runs
int64_t total_max_subarray_sum(int n, uint32_t initial_seed, int min_val, int max_val) {
    int64_t total_sum = 0;
    uint32_t seed = initial_seed;
    for (int i = 0; i < 20; ++i) {
        seed = lcg(seed, 1);
        total_sum += max_subarray_sum(n, seed, min_val, max_val);
    }
    return total_sum;
}

int main() {
    int n = 1000;
    uint32_t initial_seed = 42;
    int min_val = -10;
    int max_val = 10;

    auto start_time = chrono::high_resolution_clock::now();
    int64_t result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    auto end_time = chrono::high_resolution_clock::now();

    chrono::duration<double> execution_time = end_time - start_time;

    cout << "Total Maximum Subarray Sum (20 runs): " << result << endl;
    cout << "Execution Time: " << fixed << execution_time.count() << " seconds" << endl;

    return 0;
}