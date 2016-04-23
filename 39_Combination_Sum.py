# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:37:14 2015

@author: HSH
"""

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        allSol = []
        sol = []
        if not candidates:
            return allSol
        candidates = sorted(candidates)
        self.findCombSum(candidates, 0, target, sol, allSol)
        return allSol
        
        
    def findCombSum(self, candidates, start, target, sol, allSol):
        if target == 0:
            allSol.append(list(sol))
            return
        
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i-1]:
                continue
            if candidates[i] <= target:
                sol.append(candidates[i])
                self.findCombSum(candidates, i, target - candidates[i], sol, allSol)
                sol.pop()


sol = Solution()
sol.combinationSum([2,3,6,7],7)