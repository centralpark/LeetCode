# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:43:04 2015

@author: HSH
"""

class Solution(object):
    
    def findComb(self, n, start, k, sol, allSol):
        if not k:
            allSol.append(sol[:])
            return
        for i in range(start, n-k+2):
            sol.append(i)
            self.findComb(n, i+1, k-1, sol, allSol)
            sol.pop()
    
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        sol = []
        allSol = []
        self.findComb(n, 1, k, sol, allSol)
        return allSol
        
sol = Solution()
