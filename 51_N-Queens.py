# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:41:30 2015

@author: HSH
"""

class Solution(object):
    def validPos(self, col, irow, icol):
        for i in range(len(col)):
            if icol == col[i] or abs(irow - i) == abs(icol - col[i]):
                return False
        return True
        
    def solveNQ(self, n, irow, col, sol, allSol):
        if irow == n:
            allSol.append(sol[:])
            return
        for icol in range(n):
            if self.validPos(col, irow, icol):
                s = ['.'] * n
                s[icol] = 'Q'
                s = ''.join(s)
                sol.append(s)
                col.append(icol)
                self.solveNQ(n, irow + 1, col, sol, allSol)
                sol.pop()
                col.pop()
        return
    
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        allSol = []
        sol = []
        col = []
        self.solveNQ(n, 0, col, sol, allSol)
        return allSol
        
sol = Solution()
sol.solveNQueens(4)