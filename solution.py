# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:15:28 2015

@author: HSH
"""

import sys
import math

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        
        
class RandomListNode(object):
    def __init__(self,x):
        self.label = x
        self.next = None
        self.random = None
        
class TreeNode(object):
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None
        
        
class UndirectedGraphNode(object):
    def __init__(self, x):
        self.label = x
        self.neighbors = []
        

class MinStack(object):
    stack = []
    minStack = []    
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minStack = []

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.stack.append(x)
        if ((not self.minStack) or x <= self.minStack[-1]):
            self.minStack.append(x)

    def pop(self):
        """
        :rtype: nothing
        """
        if self.stack.pop() == self.minStack[-1]:
            self.minStack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.minStack[-1]
        

class Solution(object):
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        max_value = 2147483647
        min_value = -2147483648
        maxDiv10 = max_value/10
        i = 0
        n = len(s)
        while i<n and s[i]==' ':
            i += 1
        sign = 1
        if i<n and s[i]=='+':
            i += 1
        elif i<n and s[i]=='-':
            sign = -1
            i += 1
        num = 0
        while i<n and s[i].isdigit():
            digit = int(s[i])
            if num>maxDiv10 or (num==maxDiv10 and digit>=8):
                return max_value if sign==1 else min_value
            num = num*10+digit
            i += 1
        return sign*num
        
        
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i = 0
        n = len(s)
        while i<n and s[i]==' ':
            i += 1
        if i<n and s[i] in ['+','-']:
            i += 1
        isNumeric = False
        while i<n and s[i].isdigit():
            i += 1
            isNumeric = True
        if i<n and s[i]=='.':
            i += 1
            while i<n and s[i].isdigit():
                i += 1
                isNumeric = True
        if isNumeric and i<n and s[i]=='e':
            i += 1
            isNumeric = False
            if i<n and s[i] in ['+','-']:
                i += 1
            while i<n and s[i].isdigit():
                i += 1
                isNumeric = True
        while i<n and s[i]==' ':
            i += 1
        return isNumeric and i==n
        
        
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        div = 1
        while x/div >= 10:
            div *= 10
        while x!=0:
            l = x / div
            r = x % 10
            if l != r:
                return False
            x = (x % div) / 10
            div /= 100
        return True
    
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummyHead = ListNode(0)
        p = dummyHead
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        if l1:
            p.next = l1
        if l2:
            p.next = l2
        return dummyHead.next


    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummyHead = ListNode(0)
        dummyHead.next = head
        p = head
        prev = dummyHead
        while p and p.next:
            q = p.next
            r = p.next.next
            q.next = p
            p.next = r
            prev.next = q
            prev = p
            p = r
        return dummyHead.next
        
        
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        hashmap = dict()
        p = head
        dummy = RandomListNode(0)
        q = dummy
        while p:
            q.next = RandomListNode(p.label)
            hashmap[p] = q.next
            p = p.next
            q = q.next
        p = head
        q = dummy
        while p:
            if hashmap.has_key(p.random):
                q.next.random = hashmap[p.random]
            else:
                q.next.random = None
            p = p.next
            q = q.next
        return dummy.next
        
    
    def isSubtreeLessThan(self, p, val):
        if not p:
            return True
        return p.val < val and self.isSubtreeLessThan(p.left, val) and \
        self.isSubtreeLessThan(p.right, val)
    
    def isSubtreeGreaterThan(self, p, val):
        if not p:
            return True
        return p.val > val and self.isSubtreeGreaterThan(p.left, val) and \
        self.isSubtreeGreaterThan(p.right, val)
    
    
    prev = None
    
    def isMonotonicIncreasing(self, p):
        if not p:
            return True
        if self.isMonotonicIncreasing(p.left):
            if self.prev and p.val <= self.prev.val:
                return False
            self.prev = p
            return self.isMonotonicIncreasing(p.right)
        return False
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.prev = None
        return self.isMonotonicIncreasing(root)
        
        
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        
        
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
        
        
    def maxDepth2(self, root):
        if not root:
            return 0;
        L = self.maxDepth2(root.left)
        if L == -1:
            return -1
        R = self.maxDepth2(root.right)
        if R == -1:
            return -1
        return max(L, R) + 1 if abs(L - R) <= 1 else -1
        
        
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.maxDepth2(root) != -1
        
        
    def subArrayToBST(self, nums, start, end):
        if start > end:
            return None
        mid = (start + end)/2
        node = TreeNode(nums[mid])
        node.left = self.subArrayToBST(nums, start, mid-1)
        node.right = self.subArrayToBST(nums, mid+1, end)
        return node
        
    
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.subArrayToBST(nums, 0, len(nums)-1)
        
        
    listNode = None
    
    def subListToBST(self, start, end):
        if start > end:
            return None
        mid = (start + end) / 2
        leftChild = self.subListToBST(start, mid-1)
        parent = TreeNode(self.listNode.val)
        parent.left = leftChild
        self.listNode = self.listNode.next
        parent.right = self.subListToBST(mid+1, end)
        return parent
        
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        n = 0
        p = head
        while p:
            p = p.next
            n += 1
        self.listNode = head
        return self.subListToBST(0, n-1)
        
        
    maxSum = 0
    
    def findMax(self, p):
        if not p:
            return 0
        left = self.findMax(p.left)
        right = self.findMax(p.right)
        self.maxSum = max(p.val + left + right, self.maxSum)
        ret = p.val + max(left, right)
        return ret if ret > 0 else 0
    
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.maxSum = float("-inf")
        self.findMax(root)
        return self.maxSum
        
        
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        num = 0
        for x in nums:
            num ^= x
        return num
        
        
    def singleNumber2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        ones = 0
        twos = 0
        threes = 0
        for i in range(n):
            twos |= ones & nums[i]
            ones ^= nums[i]
            threes = ones & twos
            ones &= ~threes
            twos &= ~threes
        return ones
        
        
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        elements = []
        if len(matrix)==0:
            return elements
        m = len(matrix)
        n = len(matrix[0])
        row = 0
        col = -1
        while True:
            for i in range(n):
                col += 1
                elements.append(matrix[row][col])
            m -= 1
            if m == 0:
                break
            for i in range(m):
                row += 1
                elements.append(matrix[row][col])
            n -= 1
            if n == 0:
                break
            for i in range(n):
                col -= 1
                elements.append(matrix[row][col])
            m -= 1
            if m == 0:
                break
            for i in range(m):
                row -= 1
                elements.append(matrix[row][col])
            n -= 1
            if n == 0:
                break
        return elements
        
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 
               'IV', 'I']    
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        roman = ''
        i = 0
        while num > 0:
            k = num / self.values[i]
            for j in range(k):
                roman += self.symbols[i]
                num -= self.values[i]
            i += 1
        return roman
               
    
    hashmap = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        prev = 0
        total = 0
        for c in s:
            curr = self.hashmap[c]
            total += (curr - 2 * prev) if curr > prev else curr
            prev = curr
        return total            
    

    def DFS(self, graph, hashmap):
        if graph in hashmap:
            return hashmap[graph]
        graphCopy = UndirectedGraphNode(graph.label)
        hashmap[graph] = graphCopy
        for neighbor in graph.neighbors:
            graphCopy.neighbors.append(self.DFS(neighbor, hashmap))
        return graphCopy
        
    def cloneGraph(self, node):
        """
        :type node: UndirectedGraphNode
        :rtype: UndirectedGraphNode
        """
        if not node:
            return None
        hashmap = {}
        return self.DFS(node, hashmap)
        
        
    OPERATORS = set(["+", "-", "*", "/"])
    
    def evaluate(self, x, y, operator):
        if operator == '+':
            return x + y
        elif operator == '-':
            return  x - y
        elif operator == '*':
            return x * y
        else:
            return x / y
    
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for token in tokens:
            if token in self.OPERATORS:
                y = int(stack.pop())
                x = int(stack.pop())
                stack.append(self.evaluate(x, y, token))
            else:
                stack.append(int(token))
        return stack.pop()
        
    
    hashmap = {'(':')', '{':'}','[':']'}    
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for c in s:
            if c in self.hashmap:
                stack.append(c)
            elif (not stack) or self.hashmap[stack.pop()] != c:
                return False
        return False if stack else True
        
    
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        p = 1
        q = 1
        for i in range(2,n+1):
            temp = q
            q += p
            p = temp
        return q
        
        
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        mat = [[0] * n] * m
        mat[m - 1] = [1] * n
        for i in range(m):
            mat[i][n-1] = 1
        for r in range(m - 2, -1, -1):
            for c in range(n - 2, -1, -1):
                mat[r][c] = mat[r + 1][c] + mat[r][c + 1]
        return mat[0][0]
        
        
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if m == 0:
            return 0            
        mat = [[0 for i in range(n+1)] for j in range(m+1)]
        mat[m - 1][n] = 1
        for r in range(m - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                mat[r][c] = 0 if obstacleGrid[r][c] else mat[r + 1][c] + mat[r][c + 1]
        return mat[0][0]
        
        
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxEndingHere = nums[0]
        maxSoFar = nums[0]
        for i in range(1, len(nums)):
            maxEndingHere = max(maxEndingHere + nums[i], nums[i])
            maxSoFar = max(maxEndingHere, maxSoFar)
        return maxSoFar
        
    
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        assert len(nums) > 0
        maxEndingHere = nums[0]
        minEndingHere = nums[0]
        maxSoFar = nums[0]
        for i in range(1, len(nums)):
            mx = maxEndingHere
            maxEndingHere = max(nums[i], maxEndingHere * nums[i],
                                minEndingHere * nums[i])
            minEndingHere = min(nums[i], minEndingHere * nums[i],
                                mx * nums[i])
            maxSoFar = max(maxSoFar, maxEndingHere)
        return maxSoFar
        
        
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        L = 0
        R = len(nums) - 1
        while L < R:
            M = (L + R) / 2
            if nums[M] < target:
                L = M + 1
            else:
                R = M - 1
        return L if nums[L] >= target else L + 1
        
        
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        L = 0
        R = len(nums) - 1
        while L < R:
            M = (L + R) / 2
            if nums[M] < nums[R]:
                R = M
            else:
                L = M + 1
        return nums[L]
        
        
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        L = 0
        R = len(nums) - 1
        while L < R and nums[L] >= nums[R]:
            M = (L + R) / 2
            if nums[M] > nums[R]:
                L = M + 1
            elif nums[M] < nums[R]:
                R = M
            else:
                L = L + 1
        return nums[L]
    
    
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        """
        f[i][j]: if s[0..i-1] matches p[0..j-1]
        if p[j - 1] != '*'
            f[i][j] = f[i - 1][j - 1] && s[i - 1] == p[j - 1]
        if p[j - 1] == '*', denote p[j - 2] with x
            f[i][j] is true iff any of the following is true
            1) "x*" repeats 0 time and matches empty: f[i][j - 2]
            2) "x*" repeats >= 1 times and matches "x*x": s[i - 1] == x && f[i - 1][j]
        '.' matches any single character
        """
        m = len(s)
        n = len(p)
        dp = [[True] + [False] * m]
        for i in range(n):
            dp.append([False]*(m+1))
    
        for i in range(1, n + 1):
            x = p[i-1]
            if x == '*' and i > 1:
                dp[i][0] = dp[i-2][0]
            for j in range(1, m+1):
                if x == '*':
                    dp[i][j] = dp[i-2][j] or dp[i-1][j] or (dp[i-1][j-1] and p[i-2] == s[j-1]) or (dp[i][j-1] and p[i-2]=='.')
                elif x == '.' or x == s[j-1]:
                    dp[i][j] = dp[i-1][j-1]
    
        return dp[n][m]
        
        
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i = 0
        j = len(height) - 1
        water = 0
        while i < j:
            h = min(height[i], height[j])
            water = max(water, (j - i) * h)
            while height[i] <= h and i < j:
                i += 1
            while height[j] <= h and i < j:
                j -= 1
        return water
        
        
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        n_str = len(strs)
        if n_str == 0:
            return ''
        if n_str == 1:
            return strs[0]
        min_len = min(map(len, strs))
        prefix = ''
        for i in range(min_len):
            c = strs[0][i]
            for j in range(1, n_str):
                if strs[j][i] != c:
                    return prefix
            prefix += c
        return prefix
        
        
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        if len(nums) < 3:
            return result
        nums = sorted(nums)
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            negate = -nums[i]
            start = i + 1
            end = len(nums) - 1
            while start < end:
                if nums[start] + nums[end] == negate:
                    result.append([nums[i], nums[start], nums[end]])
                    while start < end and nums[start + 1] == nums[start]:
                        start += 1
                    while start < end and nums[end - 1] == nums[end]:
                        end -= 1
                    start += 1
                    end -= 1
                elif nums[start] + nums[end] < negate:
                    start += 1
                else:
                    end -= 1
        return result
        
        
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        import sys
        min_diff = sys.maxsize    
        result = 0
        nums = sorted(nums)
        for i in range(len(nums)):
            start = i + 1
            end = len(nums) - 1
            while start < end:
                sum_3 = nums[i] + nums[start] + nums[end]
                diff = abs(sum_3 - target)
                if sum_3 == target:
                    return target
                if diff < min_diff:
                    min_diff = diff
                    result = sum_3
                if sum_3 < target:
                    start += 1
                else:
                    end -= 1
        return result
        
        
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        keymap = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs',
                  '8':'tuv', '9':'wxyz', '0':''}
        result = []
        if len(digits) == 0:
            return result
        if len(digits) == 1:
            for c in keymap[digits]:
                result.append(c)
            return result
        result_pre = self.letterCombinations(digits[:-1])
        for s in result_pre:
            for c in keymap[digits[-1]]:
                result.append(s + c)
        return result
        
        
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        if len(nums) < 4:
            return result
        nums = sorted(nums)
        for i in range(len(nums) - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums) - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                start = j + 1
                end = len(nums) - 1
                negate = target - (nums[i] + nums[j])
                while start < end:
                    if nums[start] + nums[end] == negate:
                        result.append([nums[i], nums[j], nums[start], nums[end]])
                        while start < end and nums[start + 1] == nums[start]:
                            start += 1
                        while start < end and nums[end - 1] == nums[end]:
                            end -= 1
                        start += 1
                        end -= 1
                    elif nums[start] + nums[end] < negate:
                        start += 1
                    else:
                        end -= 1
        return result
    
    
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummyHead = ListNode(0)
        dummyHead.next = head
        i = dummyHead
        j = dummyHead
        for k in range(n):
            j = j.next
        while j.next:
            i = i.next
            j = j.next
        i.next = i.next.next
        return dummyHead.next
        
    
    def dfs(self, result, s, left, right):
        if left > right:
            return
        if left == 0 and right == 0:
            result.append(s)
            return
        if left > 0:
            self.dfs(result, s + '(', left - 1, right)
        if right > 0:
            self.dfs(result, s + ')', left, right - 1)
    
    
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        self.dfs(result, '', n, n)
        return result
        
        
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if len(lists) == 0:
            return None
        end =len(lists) - 1
        while end > 0:
            begin = 0
            while begin < end:
                lists[begin] = self.mergeTwoLists(lists[begin], lists[end])
                begin += 1
                end -= 1
        return lists[0]
        
        
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        end = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[end]:
                end += 1
                nums[end] = nums[i]
        return end + 1
    
    
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 2:
            return len(nums)
        prev = 1
        curr = 2
        while curr < len(nums):
            if nums[curr] == nums[prev] and nums[curr] == nums[prev - 1]:
                pass
            else:
                prev += 1
                nums[prev] = nums[curr]
            curr += 1
        return prev + 1
        
        
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i = 0
        j = 0
        while j < len(nums):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
            j += 1
        return i
                    
        
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        if len(board) != 9 or len(board[0]) != 9:
            return False
        # check rows
        for i in range(9):
            nums = []
            for j in range(9):
                if board[i][j] in nums:
                    return False
                if board[i][j] != '.':
                    nums.append(board[i][j])
        # check columns
        for i in range(9):
            nums = []
            for j in range(9):
                if board[j][i] in nums:
                    return False
                if board[j][i] != '.':
                    nums.append(board[j][i])
        # check each box
        for i in [0, 3, 6]:
            for j in [0, 3, 6]:
                nums = []
                for m in range(i, i+3):
                    for n in range(j, j+3):
                        if board[m][n] in nums:
                            return False
                        if board[m][n] != '.':
                            nums.append(board[m][n])
        return True
        
        
    def gcd(self, p, q):
        if q == 0:
            return p
        r = p % q
        return self.gcd(q, r)
    
    
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n == 0:
            return ''
        if n == 1:
            return '1'
        seq_pre = self.countAndSay(n - 1)
        seq = ''
        count = 1
        for i in range(1, len(seq_pre)):
            if seq_pre[i] == seq_pre[i - 1]:
                count += 1
            else:
                seq += str(count) + seq_pre[i - 1]
                count = 1
        seq += str(count) + seq_pre[-1]
        return seq
        
        
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        # handle special case
        if(divisor == 0):
            return sys.maxsize
        pDividend = abs(dividend)
        pDivisor = abs(divisor)
        result = 0
        while(pDividend >= pDivisor):
            numShift = 0
            while pDividend >= (pDivisor<<numShift):
                numShift += 1
            result += 1<<(numShift-1)
            pDividend -= (pDivisor<<(numShift-1))
        if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0):
            return result
        else:
            return -result
    
    
    
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        second = head.next
        rest = self.reverseList(head.next)
        head.next = None
        second.next = head
        return rest
        
        
    def reverseList_iterative(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        p1 = head
        p2 = head.next
        head.next = None
        while p1 and p2:
            t = p2.next
            p2.next = p1
            p1 = p2
            if t:
                p2 = t
            else:
                break
        return p2
            
    
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        newHead = None
        p2 = head
        oldTail = None
        while True:
            start = p2
            for step in range(k):
                if not start:
                    if oldTail:
                        oldTail.next = p2
                    return newHead if newHead else head
                start = start.next
            p1 = p2
            p2 = p1.next
            newTail = p1
            for step in range(k - 1):
                t = p2.next
                p2.next = p1
                p1 = p2
                p2 = t
                newTail.next = None
            if not newHead:
                newHead = p1
            if oldTail:
                oldTail.next = p1
                oldTail = newTail
            else:
                oldTail = newTail
        return newHead
        
        
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        result = []
        if len(s) ==0 or len(words) == 0:
            return result
        refMap = {}
        for w in words:
            if w in refMap:
                refMap[w] += 1
            else:
                refMap[w] = 1
        
        length = len(words[0])
        
        for j in range(length):
            currentMap = {}
            start = j
            count = 0
            
            for i in range(j, len(s) - length + 1, length):
                sub = s[i:i+length]
                if sub in refMap:
                    #set frequency in current map
                    if sub in currentMap:
                        currentMap[sub] += 1
                    else:
                        currentMap[sub] = 1
                
                    count += 1
                
                    while(currentMap[sub] > refMap[sub]):
                        left = s[start:start+length]
                        currentMap[left] -= 1
                        count -= 1
                        start += length
                    
                    if count == len(words):
                        result.append(start)
                        #shift right and reset currentMap, count & start point
                        left = s[start:start+length]
                        currentMap[left] -= 1
                        count -= 1
                        start += length
                else:
                    currentMap = {}
                    start = i + length
                    count = 0
                    
        return result
    
    
    def reverse(self, nums, left, right):
        while left < right:
            temp = nums[left]
            nums[left] = nums[right]
            nums[right] = temp
            left += 1
            right -= 1
            
    
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) < 2:
            return
        
        p = 0
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                p = i
                break
        
        q = 0
        for i in range(len(nums) - 1, p, -1):
            if nums[i] > nums[p]:
                q = i
                break
        
        if p == 0 and q == 0:
            self.reverse(nums, 0, len(nums) - 1)
            return
            
        temp = nums[p]
        nums[p] = nums[q]
        nums[q] = temp
        
        if p < len(nums) - 1:
            self.reverse(nums, p+1, len(nums)-1)
            
    
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        maxLen = 0
        curLen = 0
        for i in range(len(s)):
            if s[i] == '(': #left parenthesis
                stack.append([i,0])
            else:   #right parenthesis
                if not stack or stack[-1][1] == 1:
                    stack.append([i,1])
                else:
                    stack.pop()
                    if not stack:
                        curLen = i + 1
                    else:
                        curLen = i - stack[-1][0]
                    maxLen = max(maxLen, curLen)
        return maxLen
        
        
    def searchRotatedSortedArray(self, nums, start, end, target):
        if start > end:
            return -1
        mid = int(start + (end - start)/2)
        if nums[mid] == target:
            return mid
        if nums[mid] < nums[end]:  #right half sorted
            if target > nums[mid] and target <= nums[end]:
                return self.searchRotatedSortedArray(nums, mid + 1, end, target)
            else:
                return self.searchRotatedSortedArray(nums, start, mid - 1, target)
        else:
            if target >= nums[start] and target < nums[mid]:
                return self.searchRotatedSortedArray(nums, start, mid - 1, target)
            else:
                return self.searchRotatedSortedArray(nums, mid + 1, end, target)
    
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return self.searchRotatedSortedArray(nums, 0, len(nums) - 1, target)
        
        
    def search_v2(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = int(start + (end-start)/2)
            if nums[mid] == target:
                return True
            if nums[mid] < nums[end]:
                if target > nums[mid] and target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            elif nums[mid] > nums[end]:
                if target >= nums[start] and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                end -= 1
        return False
        
    
    def findLeftMost(self, nums, target):
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = int(start + (end-start)/2)
            if target < nums[mid]:
                end = mid - 1
            elif target > nums[mid]:
                start = mid + 1
            else:
                end = mid - 1
        if start < len(nums) and nums[start] == target:
            return start
        return -1
        
        
    def findRightMost(self, nums, target):
        start = 0
        end = len(nums) - 1
        while start <= end:
            mid = int(start + (end-start)/2)
            if target < nums[mid]:
                end = mid - 1
            elif target > nums[mid]:
                start = mid + 1
            else:
                start = mid + 1
        if end >= 0 and nums[end] == target:
            return end
        return -1
        
    
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        ran = []
        ran.append(self.findLeftMost(nums, target))
        ran.append(self.findRightMost(nums, target))
        return ran     
        
        
    def addBinary(self, a, b):
        a = a[::-1]
        b = b[::-1]
        s = []
        carry = 0
        n = max(len(a), len(b))
        for i in range(n):
            if i < len(a):
                carry += int(a[i])
            if i < len(b):
                carry += int(b[i])
            s.append(str(carry%2))
            carry = int(carry/2)
        if carry:
            s.append('1')
        return ''.join(s[::-1])
        
        
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        start = 0
        end = x
        while start <= end:
            mid = int(start + (end - start)/2)
            if x / mid == mid:
                return mid
            elif x / mid < mid:
                end = mid - 1
            else:
                start = mid + 1
        return end
        
        
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        curDir = ''
        allDir = []
        path = path + '/'
        for i in range(len(path)):
            if path[i] == '/':
                if not curDir:
                    continue
                elif curDir == '.':
                    curDir = ''
                elif curDir == '..':
                    if allDir:
                        allDir.pop()
                    curDir = ''
                else:
                    allDir.append(''.join(curDir))
                    curDir = ''
            else:
                curDir += path[i]
        if not allDir:
            return '/'
        else:
            return '/' + '/'.join(allDir)
        
        
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m = len(word1)
        n = len(word2)
        dp = [[0 for i in range(n + 1)] for j in range(m + 1)]
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            dp[i][0] = i
            for j in range(1, n + 1):
                k = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + k)
        return dp[m][n]
        

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        m = len(matrix)
        n = len(matrix[0])
        start = 0
        end = m * n - 1
        while start <= end:
            mid = int((start + end) /2 )
            midRow = int (mid / n)
            midCol = mid % n
            if matrix[midRow][midCol] == target:
                return True
            elif matrix[midRow][midCol] < target:
                start = mid + 1
            else:
                end = mid - 1
        return False
        
    
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        m = len(matrix)
        n = len(matrix[0])
        row = m - 1
        col = 0
        while row >=0 and col < n:
            if target < matrix[row][col]:
                row -= 1
            elif target > matrix[row][col]:
                col += 1
            else:
                return True
        return False
        
        
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums) - 1
        i = 0
        while i <= right:
            if nums[i] == 0:
                nums[i], nums[left] = nums[left], nums[i]
                i += 1
                left += 1
            elif nums[i] == 1:
                i += 1
            else:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
                
                
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        sLen = len(s)
        tLen = len(t)
        needToFind = {}
        hasFound = {}
        for i in range(tLen):
            if t[i] not in needToFind:
                needToFind[t[i]] = 1
            else:
                needToFind[t[i]] += 1
            hasFound[t[i]] = 0
        minWindowLen = sLen + 1
        matchLen = 0
        start = 0
        for end in range(sLen):
            if s[end] not in needToFind:
                continue
            hasFound[s[end]] += 1
            if hasFound[s[end]] <= needToFind[s[end]]:
                matchLen += 1
            if matchLen == tLen:
                while s[start] not in needToFind or hasFound[s[start]] > needToFind[s[start]]:
                    if s[start] in hasFound and hasFound[s[start]] > needToFind[s[start]]:
                        hasFound[s[start]] -= 1
                    start += 1
                windowLen = end - start + 1
                if windowLen < minWindowLen:
                    minWindowStart = start
                    minWindowEnd = end
                    minWindowLen = windowLen
        return s[minWindowStart:(minWindowEnd+1)] if matchLen == tLen else ''
            

    def largestRectangleArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if len(height) == 0:
            return 0
        stack = []
        maxArea = 0
        for i in range(len(height)):
            while stack and height[i] <= stack[-1]:
                h = height[stack[-1]]
                stack.pop()
                if height[i] < h:
                    maxArea = max(maxArea, (i - stack[-1] - 1) * h)
            stack.append(i)
        return maxArea


    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        minElement = sys.maxsize
        for i in range(len(prices)):
            profit = max(profit, prices[i] - minElement)
            minElement = min(minElement, prices[i])
        return profit
        
    def maxProfit_ii(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        profit = 0
        if n < 2:
            return profit
        for i in range(1, n):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                profit += diff
        return profit
        
        
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        import re
        if not s:
            return True
        p = re.compile('[^a-z0-9]')
        s = p.sub('', s.lower())
        start = 0
        end = len(s) - 1
        while start <= end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
        
        
    def minCut(self, s):
        n = len(s)
        if n <= 1:
            return 0
        isPal = [[False for i in range(n)] for j in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if (i + 1 > j - 1 or isPal[i + 1][j - 1]) and s[i] == s[j]:
                    isPal[i][j] = True
        dp = [sys.maxsize for i in range(n + 1)]
        dp[0] = -1
        for i in range(1, n + 1):
            for j in range(i - 1, -1, -1):
                if isPal[j][i - 1]:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n]
        
        
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        start = 0
        netGas = 0
        curGas = 0
        for i in range(len(gas)):
            netGas += gas[i] - cost[i]
            curGas += gas[i] - cost[i]
            if curGas < 0:
                start = i + 1
                curGas = 0
        if netGas < 0:
            return -1
        return start
    
    
    # 135 Candy
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        n = len(ratings)
        candies = [0 for i in range(n)]
        candies[0] = 1
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
            else:
                candies[i] = 1
        for i in range(n - 2, -1, -1):
            cur = 1
            if ratings[i] > ratings[i + 1]:
                cur = candies[i + 1] + 1
            candies[i] = max(candies[i], cur)
        return sum(candies)
        
        
    def wordBreak(self, s, wordDict):
        n = len(s)
        dp = [False for i in range(n + 1)]
        dp[0] = True
        for i in range(n):
            for j in range(i, -1, -1):
                if dp[j] and s[j:(i + 1)] in wordDict:
                    dp[i + 1] = True
                    break
        return dp[n]
        
        
    def maximumGap(self,nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return 0
        # get the max and min values of the array
        mini = float('inf')
        maxi = -float('inf')
        for num in nums:
            mini = min(mini, num)
            maxi = max(maxi, num)
        gap = math.ceil((float(maxi) - float(mini))/(len(nums) - 1))
        bucketsMin = [float('inf')] * (len(nums) - 1)
        bucketsMax = [float('-inf')] * (len(nums) - 1)
        for i in nums:
            if i == mini or i == maxi:
                continue
            idx = int((i - mini) / gap)
            bucketsMin[idx] = min(i, bucketsMin[idx])
            bucketsMax[idx] = max(i, bucketsMax[idx])
        maxGap = -float('inf')
        previous = mini
        for i in range(len(nums) - 1):
            if bucketsMin[i] == float('inf') and bucketsMax[i] == -float('inf'):
                continue
            maxGap = max(maxGap, bucketsMin[i] - previous)
            previous = bucketsMax[i]
        maxGap = max(maxGap, maxi - previous)
        return maxGap
    
            
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1 = version1.split('.')
        v2 = version2.split('.')
        length = max(len(v1),len(v2))
        for i in range(length):
            i1 = int(v1[i]) if i < len(v1) else 0
            i2 = int(v2[i]) if i < len(v2) else 0
            if i1 > i2:
                return 1
            elif i1 < i2:
                return -1
            else:
                continue
        return 0
        
        
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if numerator == 0:
            return '0';
        res = ''
        if (numerator < 0) ^ (denominator < 0):
            res += '-'
        n = abs(numerator)
        d = abs(denominator)
        res += str(n//d)
        if n % d == 0:
            return res
        res += '.'
        m = dict()
        r = n % d        
        while True:
            if r in m:
                res = res[:m[r]] + '(' + res[m[r]:]
                res += ')'
                break
            m[r] = len(res)
            r *= 10
            res += str(r//d)
            r = r % d
            if r == 0:
                break
        return res
                
        
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        count = {}
        for i in range(len(s) - 9):
            subs = s[i:(i+10)]
            if subs in count:
                count[subs] = count[subs] + 1
            else:
                count[subs] = 1
        result = []
        for subs in count:
            if count[subs] > 1:
                result.append(subs)
        return result
        
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        size = len(nums)
        if size < 0:
            return 0
        i, j = 0, 0
        result = size
        sum = nums[0]
        while j < size:
            if i==j:
                if nums[i] >= s:
                    return 1
                else:
                    j += 1
                    if j < size:
                        sum += nums[j]
                    else:
                        if i == 0:
                            return 0
            else:
                if sum >= s:
                    result = min(j-i+1, result)
                    sum -= nums[i]                    
                    i += 1
                else:
                    j += 1
                    if j < size:
                        sum += nums[j]
                    else:
                        if i == 0:
                            return 0
        return result
        
if __name__ == '__main__':
    sol = Solution()
    result = sol.minSubArrayLen(11,[1,2,3,4,5])
    print(result)