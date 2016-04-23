# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:40:14 2015

@author: HSH
"""

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        allSet = []
        sol = []
        allSet.append(sol[:])
        nums = sorted(nums)
        self.findSubsets(nums, 0, sol, allSet)
        return allSet
        
    def findSubsets(self, nums, start, sol, allSet):
        for i in range(start, len(nums)):
            sol.append(nums[i])
            allSet.append(sol[:])
            self.findSubsets(nums, i + 1, sol, allSet)
            sol.pop()
