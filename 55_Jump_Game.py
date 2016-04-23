# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:48:47 2015

@author: HSH
"""

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        maxIndex = 0
        for i in range(n):
            if i > maxIndex or maxIndex >= (n - 1):
                break
            maxIndex = max(maxIndex, nums[i] + i)
        return True if maxIndex >= n-1 else False