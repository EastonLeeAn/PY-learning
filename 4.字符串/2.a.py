# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     2.a
   Description :
   Author :       lizhenhui
   date：          2024/6/17
-------------------------------------------------
   Change Activity:
                   2024/6/17:
-------------------------------------------------
"""
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        if len(s) == 0:
            return False
        nxt = [0] * len(s)
        self.getNext(nxt, s)
        if nxt[-1] != -1 and len(s) % (len(s) - (nxt[-1] + 1)) == 0:
            return True
        return False