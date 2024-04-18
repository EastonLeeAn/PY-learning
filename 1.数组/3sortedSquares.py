# -*- coding: utf-8 -*-
"""
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]，排序后，数组变为 [0,1,9,16,100]
"""
# 暴力解法就是 平方之后快排
# 时间复杂度是O(n + nlogn)


# 双指针办法---有序数组

def sortedSquares(nums:list) -> list:
    l=0
    r=len(nums)
    i=len(nums)


    res = [float('inf')] * len(nums)
    while l <= r :
        if nums[l]*nums[l] < nums[r]*nums[r]:
            res[i] = nums[r] ** 2
            r -= 1
        else:
            res[i] = nums[l] ** 2
            l += 1

        i -= 1
    return res

print([float('inf')] * 3)
print(float('inf'))