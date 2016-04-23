# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:19:18 2015

@author: HSH
"""

class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        result = [[0 for x in range(n)] for x in range(n)]
        nlevel = int(n/2)
        val = 1
        for i in range(nlevel):
            last = n - 1 - i
            for j in range(i,last):
                result[i][j] = val
                val += 1
            for j in range(i, last):
                result[j][last] = val
                val += 1
            for j in range(last, i, -1):
                result[last][j] = val
                val += 1
            for j in range(last, i, -1):
                result[j][i] = val
                val += 1
        if n%2 == 1:
            result[nlevel][nlevel] = val
        return result   