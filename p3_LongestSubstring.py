class Solution(object):
    def lengthOfLongestSubstring(self,s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return 0
        maxLen = 0
        chars = {}
        j = 0
        for i in range(len(s)):
            if s[i] in chars:
                j = max(j,chars[s[i]]+1)
            chars[s[i]] = i
            maxLen = max(maxLen,i-j+1)
        return maxLen
