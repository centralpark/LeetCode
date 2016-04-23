# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:54:04 2015

@author: HSH
"""

class Solution(object):
    def isValid(self, board, irow, icol):
        val = board[irow][icol]
        # check row & col
        for i in range(9):
            if board[irow][i] == val and i != icol:
                return False
            if board[i][icol] == val and i != irow:
                return False
        irow0 = int(irow/3) * 3
        icol0 = int(icol/3) * 3
        for i in range(irow0, irow0+3):
            for j in range(icol0, icol0+3):
                if board[i][j] == val and (i != irow or j != icol):
                    return False
        return True
        
    
    def solSudoku(self, board, row, col):
        if row == 9:
            return True
        if col == 8:
            row2 = row + 1
            col2 = 0
        else:
            row2 = row
            col2 = col + 1
        
        if board[row][col] != '.':
            if not self.isValid(board, row, col):
                return False
            return self.solSudoku(board, row2, col2)
        for i in range(1,10):
            board[row][col] = str(i)
            if self.isValid(board, row, col) and self.solSudoku(board, row2, col2):
                return True
        board[row][col] = '.'
        return False
        
            
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if len(board) < 9 or len(board[0]) < 9:
            return
        for i in range(9):
            board[i] = list(board[i])
        self.solSudoku(board, 0, 0)
        for i in range(9):
            board[i] = ''.join(board[i])                


sol = Solution()
tmp = ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
sol.solveSudoku(tmp)