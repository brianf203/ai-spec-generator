"""Knapsack."""
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i]-1, -1):
            dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[capacity]
def knapsack_unbounded(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w in range(1, capacity+1):
        for i in range(len(weights)):
            if weights[i] <= w: dp[w] = max(dp[w], dp[w-weights[i]] + values[i])
    return dp[capacity]
def subset_sum(arr, target):
    dp = [False] * (target + 1)
    dp[0] = True
    for x in arr:
        for t in range(target, x-1, -1): dp[t] = dp[t] or dp[t-x]
    return dp[target]
def count_subset_sum(arr, target):
    dp = [0] * (target + 1)
    dp[0] = 1
    for x in arr:
        for t in range(target, x-1, -1): dp[t] += dp[t-x]
    return dp[target]