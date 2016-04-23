# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:56:46 2016

@author: HSH
"""

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        n = len(s)
        dp = [[] for i in range(n + 1)]
        dp[0] = True
        for i in range(n):
            if not dp[i]:
                continue
            for j in range(i, n):
                word = s[i:(j + 1)]
                if word in wordDict:
                    dp[j + 1].append(word)
        dp[0] = []
        result = []
        if not dp[n]:
            return result
        res = []
        self.dfs(dp, n, result, res)
        return result
                
    def dfs(self, dp, end, result, res):
        if end == 0:
            result.append(' '.join(res[::-1]))
            return
        for s in dp[end]:
            res.append(s)
            self.dfs(dp, end - len(s), result, res)
            res.pop()
            
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return
        slow = head
        fast = head
        if not fast.next or not fast.next.next:
            return
        fast = fast.next.next
        slow = slow.next
        while slow != fast:
            if not fast.next or not fast.next.next:
                return
            fast = fast.next.next
            slow = slow.next
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
            
sol = Solution()
result = sol.wordBreak("catsanddog",set(["cat", "cats", "and", "sand", "dog"]))