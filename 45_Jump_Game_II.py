# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:02:18 2015

@author: HSH
"""

class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        njump = 0
        curMax = 0
        lastLastMax = -1
        i = 0
        while curMax < len(nums) - 1:
            lastMax = curMax
            for i in range(lastLastMax + 1, lastMax + 1):
                curMax = max(curMax, nums[i] + i)
            if curMax == lastMax:
                return -1
            lastLastMax = lastMax
            njump += 1
        return njump
        
sol = Solution()
sol.jump([1,2])