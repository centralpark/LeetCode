# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:15:31 2015

@author: HSH
"""

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        hashmap = {}
        allAnagram = []
        for i in range(len(strs)):
            s = ''.join(sorted(strs[i]))
            if s in hashmap:
                hashmap[s].append(i)
            else:
                hashmap[s] = [i]
        for s in hashmap:
            anagram = []
            for i in hashmap[s]:
                anagram.append(strs[i])
            anagram = sorted(anagram)
            allAnagram.append(anagram)
        return allAnagram
        
sol = Solution()
sol.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])