# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     reverseList
   Description :
   Author :       lizhenhui
   date：          2024/3/21
-------------------------------------------------
   Change Activity:
                   2024/3/21:
-------------------------------------------------
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:  ##双指针法
    def reverseList(self, head: ListNode) -> ListNode:
        cur = head
        pre = None
        while cur:
            temp = cur.next # 保存一下 cur的下一个节点，因为接下来要改变cur->next
            cur.next = pre #反转

            # cur.next指的是那个箭头 往前面指

            #更新pre、cur指针
            pre = cur
            cur = temp
        return pre


class Solution: ##di gui
    def reverseList(self, head: ListNode) -> ListNode:
        return self.reverse(head, None)

    def reverse(self, cur: ListNode, pre: ListNode) -> ListNode:
        if cur == None:
            return pre
        temp = cur.next # 保存一下 cur的下一个节点，因为接下来要改变cur->next
        cur.next = pre  # cur.next 指的是那个箭头   指向pre 当前节点 变成空
        return self.reverse(temp, cur)