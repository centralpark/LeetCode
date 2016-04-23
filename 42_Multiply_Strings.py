# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:39:11 2015

@author: HSH
"""

class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if not num1 or not num2:
            return ''
        num1 = num1[::-1]
        num2 = num2[::-1]
        result = ['0'] * (len(num1) + len(num2))
        
        for j in range(len(num2)):
            carry = 0
            val = int(num2[j])
            for i in range(len(num1)):
                carry += int(num1[i]) * val + int(result[i+j])
                result[i+j] = str(carry % 10)
                carry = int(carry/10)
            if carry != 0:
                result[len(num1) + j] = str(carry)
        result.reverse()
        
        count = 0
        while count < len(result) - 1 and result[count] == '0':
            count += 1
        return ''.join(result[count:])