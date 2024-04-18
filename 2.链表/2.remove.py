# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     2.remove
   Description :
   Author :       lizhenhui
   date：          2024/3/20
-------------------------------------------------
   Change Activity:
                   2024/3/20:
-------------------------------------------------
"""
from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkList:
    # 新建一个头指针为空的链表

    def __init__(self):
        self.head = None

    def initList(self, data):
        # 创建头结点
        self.head = ListNode(data[0])
        head = self.head  # ListNode 类 （head类）
        tail = self.head  # ListNode 类
        # 逐个为 data 内的数据创建结点, 建立链表
        for i in data[1:]:  ###   尾插法
            node = ListNode(i)  # node 新类
            tail.next = node  # tail类 里边的  next 等于 node（也是类）
            tail = node     #tail  变成最后一个类
        return head

    def printlist(self, head):
        if head == None: return
        node = head
        while node != None:
            print(node.val, end=',')
            node = node.next

        print()


class Solution:

    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # 创建虚拟头部节点以简化删除过程，让头部虚节点变成第一个
        dummy_head = ListNode(next=head)

        # 遍历列表并删除值为val的节点
        current = dummy_head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next  #将（要删除的）下一个链接到 （要杉树的前一个）
            else:
                current = current.next

        return dummy_head.next


head11 = [1, 2, 6, 3, 4, 5, 6]
val1 = 6
headaaa = LinkList()
a = headaaa.initList(head11)
headaaa.printlist(a)
slu = Solution()
q = slu.removeElements(a, 6)

headaaa.printlist(q)
