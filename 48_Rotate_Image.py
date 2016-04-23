# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:51:34 2015

@author: HSH
"""

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        start = 0
        end = len(matrix) - 1
        while start < end:
            for i in range(start, end):
                offset = i - start
                temp = matrix[start][i]
                matrix[start][i] = matrix[end - offset][start]
                matrix[end - offset][start] = matrix[end][end - offset]
                matrix[end][end - offset] = matrix[start + offset][end]
                matrix[start + offset][end] = temp
            start += 1
            end -= 1
        return