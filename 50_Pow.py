# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:03:40 2015

@author: HSH
"""

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            n = -n
            x = 1/x
        return self.myPow(x * x, int(n/2)) if n%2 == 0 else x*self.myPow(x * x, int(n/2))