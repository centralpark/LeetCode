# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:53:43 2015

@author: HSH
"""

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k < 1 or not head:
            return head
        p = head
        length = 1
        while p.next:
            p = p.next
            length += 1
        p.next = head
        k = length - k % length
        while k > 0:
            p = p.next
            k -= 1
        head = p.next
        p.next = None
        return head