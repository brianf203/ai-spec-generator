"""Longest common subsequence."""
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
def lcs_length(a, b):
    """Return length of longest common subsequence."""
    return lcs(a, b)
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
def lps(s):
    n = len(s)
    if n == 0: return 0
    dp = [[0]*n for _ in range(n)]
    for i in range(n): dp[i][i] = 1
    for L in range(2, n+1):
        for i in range(n-L+1):
            j = i + L - 1
            if s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2 if L > 2 else 2
            else: dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]