# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 11:26:56 2015

@author: HSH
"""

class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.value = ''
        self.children = [None] * 26

class Trie(object):

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        p = self.root
        for c in word:
            ind = ord(c) - ord('a')
            if not p.children[ind]:
                p.children[ind] = TrieNode()
            p = p.children[ind]
        p.value = word

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.root
        for c in word:
            ind = ord(c) - ord('a')
            if not p.children[ind]:
                return False
            p = p.children[ind]
        return True if p.value == word else False
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        p = self.root
        for c in prefix:
            ind = ord(c) - ord('a')
            if not p.children[ind]:
                return False
            p = p.children[ind]
        return True
        
        
class Solution(object):
    result = []
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        trie = Trie()
        for word in words:
            trie.insert(word)
        m = len(board)
        n = len(board[0])
        visited = [[False for i in range(n)] for j in range(m)]
        for i in range(m):
            for j in range(n):
                self.dfs(board, visited, '', i, j, trie)
        return self.result
        
        
    def dfs(self, board, visited, string, row, col, trie):
        m = len(board)
        n = len(board[0])
        if row < 0 or col < 0 or row >=m or col >= n or visited[row][col]:
            return
        string += board[row][col]
        if not trie.startsWith(string):
            return
        if trie.search(string):
            self.result.append(string)
        visited[row][col] = True
        self.dfs(board, visited, string, row-1, col, trie)
        self.dfs(board, visited, string, row+1, col, trie)
        self.dfs(board, visited, string, row, col-1, trie)
        self.dfs(board, visited, string, row, col+1, trie)
        visited[row][col] = False
