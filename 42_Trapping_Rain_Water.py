# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:45:14 2015

@author: HSH
"""

class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n < 3:
            return 0
        leftHeight = [0] * n
        rightHeight = [0] * n
        water = 0
        for i in range(1,n):
            leftHeight[i] = max(leftHeight[i-1], height[i-1])
        for i in range(n-2, 0, -1):
            rightHeight[i] = max(rightHeight[i+1], height[i+1])
            minHeight = min(leftHeight[i],rightHeight[i])
            if minHeight > height[i]:
                water += minHeight - height[i]
        return water
        
sol = Solution()
sol.trap([4,2,0,3,2,5])