# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:08:21 2016

@author: HSH
"""

class Solution(object):
    def partition(self, s):
        n = len(s)
        res = []
        sol = []
        isPalin = [[False for i in range(n)] for j in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (i+1 >= j-1 or isPalin[i+1][j-1]) and s[i] == s[j]:
                    isPalin[i][j] = True
        self.findPartitions(s, 0, isPalin, sol, res)
        return res
    
    
    def findPartitions(self, s, start, isPalin, sol, res):
        if start == len(s):
            res.append(sol[:])
            return
        for i in range(start, len(s)):
            if isPalin[start][i]:
                sol.append(s[start:(i+1)])
                self.findPartitions(s, i+1, isPalin, sol, res)
                sol.pop()


if __name__ == '__main__':
    sol = Solution()
    sol.partition('a')