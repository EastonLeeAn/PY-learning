# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     isHappy
   Description :
   Author :       lizhenhui
   date：          2024/3/22
-------------------------------------------------
   Change Activity:
                   2024/3/22:
-------------------------------------------------
"""


class Solution:
    def isHappy(self, n: int) -> bool:
        record = set()

        while True:
            n = self.get_sum(n)
            if n == 1:
                return True

            # 如果中间结果重复出现，说明陷入死循环了，该数不是快乐数
            if n in record:
                return False
            else:
                record.add(n)

    def get_sum(self,n: int) -> int:
        new_num = 0
        while n:
            n, r = divmod(n, 10)
            new_num += r ** 2
        return new_num

###数字变成字符串，然后 分别相乘后相加

class Solution:
   def isHappy(self, n: int) -> bool:


       record = set()
       while n not in record:
           record.add(n)
           new_num = 0
           n_str = str(n)
           for i in n_str:
               new_num+=int(i)**2
           if new_num==1: return True
           else: n = new_num
       return False

###使用数组

class Solution:
   def isHappy(self, n: int) -> bool:
       record = []
       while n not in record:
           record.append(n)
           new_num = 0
           n_str = str(n)
           for i in n_str:
               new_num+=int(i)**2
           if new_num==1: return True
           else: n = new_num
       return False

a=Solution()
print(a.isHappy(45665))

