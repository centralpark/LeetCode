# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:26:18 2016

@author: HSH
"""

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        mid = self.findMid(head)
        right = self.reverseList(mid)
        left = head
        
        while right and right.next:
            target = right
            right = right.next
            target.next = left.next
            left.next = target
            left = left.next.next
        
    def findMid(self, head):
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
        
    def reverseList(self, head):
        if not head:
            return head
        pre = head
        cur = head.next
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        head.next = None
        return pre
            