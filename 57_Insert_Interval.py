# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:05:17 2015

@author: HSH
"""


class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
        
        
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        result = []
        isInsert = False
        for i in range(len(intervals)):
            # newInterval already inserted
            if isInsert:
                result.append(intervals[i])
                continue
            # insert newInterval before current interval
            if newInterval.end < intervals[i].start:
                result.append(newInterval)
                result.append(intervals[i])
                isInsert = True
                continue
            # combine newInterval with current interval
            if newInterval.start <= intervals[i].end:
                newInterval.start = min(newInterval.start, intervals[i].start)
                newInterval.end = max(newInterval.end, intervals[i].end)
                continue
            result.append(intervals[i])
        if not isInsert:
            result.append(newInterval)
        return result