# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:10:58 2015

@author: HSH
"""

class Solution(object):
    nSol = 0
    def validPos(self, col, irow, icol):
        for i in range(len(col)):
            if icol == col[i] or abs(irow - i) == abs(icol - col[i]):
                return False
        return True
        
    def solveNQ(self, n, irow, col):
        if irow == n:
            self.nSol += 1
            return
        for icol in range(n):
            if self.validPos(col, irow, icol):
                col.append(icol)
                self.solveNQ(n, irow + 1, col)
                col.pop()
        return
    
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        col = []
        self.solveNQ(n, 0, col)
        return self.nSol
        
sol = Solution()
sol.totalNQueens(8)