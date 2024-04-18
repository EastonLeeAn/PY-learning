# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     swapPairs
   Description :
   Author :       lizhenhui
   date：          2024/3/21
-------------------------------------------------
   Change Activity:
                   2024/3/21:
-------------------------------------------------
"""
"""给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy_head = ListNode(next=head)
        current = dummy_head
        while current.next and current.next.next:
            temp = current.next  # 防止节点修改
            temp1 = current.next.next.next

            current.next = current.next.next #更新current
            current.next.next = temp
            current.next.next.next = temp1

            current = current.next.next
        return dummy_head.next

class Solution:
    def swaplist(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        pre = head
        current = head.next
        next = head.next.next

        current.next = pre # 交换 current 是头节点

        pre.next = self.swaplist(next)
        return  current





