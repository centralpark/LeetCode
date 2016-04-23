# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:28:51 2015

@author: HSH
"""

class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        result = []
        factorial = [1 for i in range(n)]
        num = list(range(1, n + 1))
        for i in range(1, n):
            factorial[i] = factorial[i - 1] * i
        k -= 1
        for i in range(n, 0, -1):
            j = int(k/factorial[i - 1])
            k %= factorial[i - 1]
            result.append(str(num[j]))
            del num[j]
        return ''.join(result)