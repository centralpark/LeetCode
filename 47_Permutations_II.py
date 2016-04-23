
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:11:42 2015

@author: HSH
"""

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        allPer = []
        if not nums:
            return allPer
        lastPer = [[nums[0]]]
        for i in range(1, len(nums)):
            for per in lastPer:
                for j in range(len(per) + 1):
                    newPer = per[0:j] + [nums[i]]+ per[j:]
                    if newPer in allPer:
                        continue
                    allPer.append(newPer)
            lastPer = allPer
            allPer = []
        return lastPer
        
    
sol = Solution()
sol.permuteUnique([1,1,2])