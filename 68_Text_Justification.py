# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:37:25 2015

@author: HSH
"""

class Solution(object):
    def createLine(self, words, L, start, end, totLen, isLast):
        result = []
        result.append(words[start])
        n = end - start + 1
        if n == 1 or isLast:
            for i in range(start+1, end+1):
                result.append(' ')
                result.append(words[i])
            j = L - totLen - (n - 1)
            result.append(' '*j)
            return ''.join(result)
        k = int((L - totLen) / (n - 1))
        m = (L - totLen) % (n - 1)
        for i in range(start + 1, end+1):
            nspace = k + 1 if i - start <=m else k
            result.append(' ' * nspace)
            result.append(words[i])
        return ''.join(result)
        
    
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        start = 0
        end = -1
        totLen = 0
        result = []
        i = 0
        while i < len(words):
            newLen = totLen + (end - start + 1) + len(words[i])
            if newLen <= maxWidth:
                end = i
                totLen += len(words[i])
                i += 1
            else:
                line = self.createLine(words, maxWidth, start, end, totLen, False)
                result.append(line)
                start = i
                end = i - 1
                totLen = 0
        lastLine = self.createLine(words, maxWidth, start, end, totLen, True)
        result.append(lastLine)
        return result