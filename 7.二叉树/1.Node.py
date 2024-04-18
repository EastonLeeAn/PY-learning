# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     1.Node
   Description :
   Author :       lizhenhui
   date：          2024/3/24
-------------------------------------------------
   Change Activity:
                   2024/3/24:
-------------------------------------------------
"""

#类似链表 不过分为左边分支，和右边的分支

class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right