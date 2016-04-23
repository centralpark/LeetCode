# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:56:29 2015

@author: HSH
"""

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals = sorted(intervals, key = lambda inter: inter.start)
        result = []
        for i in range(len(intervals)):
            if not result or intervals[i].start > result[-1].end:
                result.append(intervals[i])
            else:
                result[-1].end = max(result[-1].end, intervals[i].end)
        return result