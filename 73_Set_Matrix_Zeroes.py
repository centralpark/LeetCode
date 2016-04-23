# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:53:52 2015

@author: HSH
"""

class Solution(object):
    def setRowCol(self, matrix, index, setRow):
        m = len(matrix)
        n = len(matrix[0])
        if setRow:
            for j in range(n):
                matrix[index][j] = 0
        else:
            for i in range(m):
                matrix[i][index] = 0
                
    
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        if m == 0:
            return
        n = len(matrix[0])
        if n == 0:
            return
        isFirstRowZero = False
        isFirstColZero = False
        for j in range(n):
            if matrix[0][j] == 0:
                isFirstRowZero = True
                break
        for i in range(m):
            if matrix[i][0] == 0:
                isFirstColZero = True
                break
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, m):
            if matrix[i][0] == 0:
                self.setRowCol(matrix, i, True)
        for j in range(1, n):
            if matrix[0][j] == 0:
                self.setRowCol(matrix, j, False)
        if isFirstRowZero:
            self.setRowCol(matrix, 0, True)
        if isFirstColZero:
            self.setRowCol(matrix, 0, False)