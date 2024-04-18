from typing import List


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        nums = [[0] * n for _ in range(n)]  # _ 临时变量，只使用一次  //初始化矩阵
        startx, starty = 0, 0  # 起始点 初始化起点
        count = 1
        offset = 1
        loop = n//2
        mid = n//2
        for offset in range(1, loop + 1):
            for i in range(starty,n-offset):
                nums[startx][i] = count
                count += 1
            for i in range(startx,n-offset):
                #//nums[i][starty] = count #//是第n-offset列
                nums[i][n-offset] = count
                count += 1
            for i in range(n-offset,starty,-1):
                nums[n-offset][i] = count
                count += 1
            for i in range(n-offset,startx,-1):
                nums[i][starty] = count
                count += 1
            startx += 1  # 更新起始点
            starty += 1
        if n % 2 != 0:  # n为奇数时，填充中心点
            nums[mid][mid] = count
        return nums













        startx, starty = 0, 0  # 起始点
        loop, mid = n // 2, n // 2  # 迭代次数、n为奇数时，矩阵的中心点
        count = 1  # 计数

        for offset in range(1, loop + 1):  # 每循环一层偏移量加1，偏移量从1开始
            for i in range(starty, n - offset):  # 从左至右，左闭右开
                nums[startx][i] = count
                count += 1
            for i in range(startx, n - offset):  # 从上至下
                nums[i][n - offset] = count
                count += 1
            for i in range(n - offset, starty, -1):  # 从右至左
                nums[n - offset][i] = count
                count += 1
            for i in range(n - offset, startx, -1):  # 从下至上
                nums[i][starty] = count
                count += 1
            startx += 1  # 更新起始点
            starty += 1

        if n % 2 != 0:  # n为奇数时，填充中心点
            nums[mid][mid] = count
        return nums


a = Solution()
aa = a.generateMatrix(3)
print(aa)
