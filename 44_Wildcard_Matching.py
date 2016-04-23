# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:37:22 2015

@author: HSH
"""

"""
# DP solution
# Time Limite Exceeded

class Solution(object):
    
"""

class Solution:
    # DP solution
    # Time Limite Exceeded
    def isMatch_v1(self, s, p):
        m = len(s)
        n = len(p)
        dp = [[False] * (m + 1)]
        for k in range(n):
            dp.append([False] * (m + 1))
        dp[0][0] = True
        for i in range(1, n + 1):
            if p[i - 1] == '*':
                dp[i][0] = dp[i - 1][0]
            for j in range(1, m + 1):
                if p[i - 1] == s[j - 1] or p[i - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[i - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j] or dp[i - 1][j - 1]
        return dp[n][m]
        
    
    def isMatch(self, s, p):
        s_cur = 0
        p_cur = 0
        match = 0
        star = -1
        while s_cur < len(s):
            if p_cur < len(p) and (s[s_cur] == p[p_cur] or p[p_cur] == '?'):
                s_cur += 1
                p_cur += 1
            elif p_cur < len(p) and p[p_cur] == '*':
                match = s_cur
                star = p_cur
                p_cur += 1
            elif star != -1:
                p_cur = star + 1
                match = match + 1
                s_cur = match
            else:
                return False
        while p_cur < len(p) and p[p_cur] == '*':
            p_cur += 1
        if p_cur == len(p):
            return True
        else:
            return False
        
        
sol = Solution()
sol.isMatch("abedd", "?b*d")
                