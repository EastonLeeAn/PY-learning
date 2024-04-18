# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     8.detectCycle
   Description :
   Author :       lizhenhui
   date：          2024/3/21
-------------------------------------------------
   Change Activity:
                   2024/3/21:
-------------------------------------------------
"""
"""题意： 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。"""
# （版本一）快慢指针法


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            # If there is a cycle, the slow and fast pointers will eventually meet
            if slow == fast:
                # Move one of the pointers back to the start of the list
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        # If there is no cycle, return None
        return None

# （版本二）集合法


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        visited = set()

        while head:
            if head in visited:
                return head
            visited.add(head)
            head = head.next

        return None