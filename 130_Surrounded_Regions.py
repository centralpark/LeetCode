# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:48:37 2016

@author: HSH
"""

class Solution(object):
    import queue as Q   
    queue = Q.Queue()
    def solve(self, board):
        if len(board) < 3 or len(board[0]) < 3:
            return
        m = len(board)
        n = len(board[0])
        # merge O's on left & right boarder        
        for i in range(m):
            if board[i][0] == 'O':
                self.bfs(board, i, 0)
            if board[i][n-1] == 'O':
                self.bfs(board, i, n-1)
        # merge O's on top & bottom boarder
        for j in range(1, n-1):
            if board[0][j] == 'O':
                self.bfs(board, 0, j)
            if board[m-1][j] == 'O':
                self.bfs(board, m - 1, j)
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == '#':
                    board[i][j] = 'O'
        
    
    def bfs(self, board, i, j):
        n = len(board[0])
        # filll current first and then its neighbors
        self.fillCell(board, i, j)
        while not self.queue.empty():
            cur = self.queue.get()
            x = cur // n
            y = cur % n
            self.fillCell(board, x - 1, y)
            self.fillCell(board, x + 1, y)
            self.fillCell(board, x, y - 1)
            self.fillCell(board, x, y + 1)
    
    
    def fillCell(self, board, i, j):
        m = len(board)
        n = len(board[0])
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
            return
        # add current cell in queue & then process its neighbors in bfs
        self.queue.put(i * n + j)
        board[i][j] = '#'
        