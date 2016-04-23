# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:46:52 2015

@author: HSH
"""

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        L = len(haystack)
        if len(needle)==0:
            return 0
        if L == 0:
            return -1
        for i in range(L+1):
            for j in range(len(needle)+1):
                if j==len(needle):
                    return i
                if (i+j)==L:
                    return -1
                if needle[j] != haystack[i+j]:
                    break

                
if __name__ == '__main__':
    sol = Solution()
    result = sol.strStr('mississippi','a')
    print result