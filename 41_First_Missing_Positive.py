# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:12:47 2015

@author: HSH
"""

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        n = len(nums)
        while i < n:
            if nums[i] != i+1 and nums[i] > 0 and nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                temp = nums[i]
                nums[i] = nums[nums[i] - 1]
                nums[temp - 1] = temp                
            else:
                i += 1
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1
        
sol = Solution()
sol.firstMissingPositive([1,1])