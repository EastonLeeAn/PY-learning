# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     7.repeatedSubstringPattern
   Description :
   Author :       lizhenhui
   date：          2024/6/17
-------------------------------------------------
   Change Activity:
                   2024/6/17:
-------------------------------------------------
"""


class Solution:
    def repeated1(self, strs):
        t = strs + strs
        t = t[1:-1]
        if strs in t:
            return True  # 如果s是t的子串
        return False

    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)
        if n <= 1:
            return False

        substr = ""
        for i in range(1, n // 2 + 1):
            if n % i == 0:
                substr = s[:i]
                if substr * (n // i) == s:
                    return True

        return False


## KMP 算法
class Solution1:
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != -1 and len(s) % (len(s) - (nxt[-1] + 1)) == 0:
            return True
        return False

    def getNext(self, nxt, s):
        nxt[0] = -1
        j = -1
        for i in range(1, len(s)):
            while j >= 0 and s[i] != s[j + 1]:
                j = nxt[j]
            if s[i] == s[j + 1]:
                j += 1
            nxt[i] = j
        return nxt


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != 0 and len(s) % (len(s) - nxt[-1]) == 0:
            return True
        return False

    def getNext(self, nxt, s):
        nxt[0] = 0
        j = 0
        for i in range(1, len(s)):
            while j > 0 and s[i] != s[j]:
                j = nxt[j - 1]
            if s[i] == s[j]:
                j += 1
            nxt[i] = j
        return nxt