# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:10:52 2015

@author: HSH
"""

class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid) == 0:
            return 0
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for i in range(n)] for j in range(m)]
        dp[0][0] = grid[0][0]
        # initialize top row
        for i in range(1, n):
            dp[0][i] = dp[0][i - 1] + grid[0][i]
        # initialize left column
        for j in range(1, m):
            dp[j][0] = dp[j - 1][0] + grid[j][0]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 1] if dp[i - 1][j] > dp[i][j - 1] else dp[i - 1][j]
                dp[i][j] += grid[i][j]
        return dp[m - 1][n - 1]