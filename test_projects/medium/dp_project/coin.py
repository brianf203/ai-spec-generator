"""Coin change."""
def coin_change(coins, amount):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] = min(dp[i], dp[i - c] + 1)
    if dp[amount] != float("inf"):
        return dp[amount]
    return -1
def coin_change_ways(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for c in coins:
        for i in range(c, amount + 1):
            dp[i] += dp[i - c]
    return dp[amount]
def coin_change_min_coins(coins, amount):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if i >= c:
                dp[i] = min(dp[i], dp[i - c] + 1)
    if dp[amount] != float("inf"):
        return dp[amount]
    return -1