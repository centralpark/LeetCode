# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 13:09:16 2015

@author: HSH
"""

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        words = s.split()
        return len(words[-1]) if len(words) > 0 else 0        