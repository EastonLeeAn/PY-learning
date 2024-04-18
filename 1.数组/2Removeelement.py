
class Solution:
    def removeElement(self, nums: list[int], val: int) -> int:
        #双循环暴力求解
        i, l =0 ,len(nums)
        while i < l:
            if nums[i] == val:
                for j in range (i+1,l):
                    nums[j-1] = nums[j] # 移除元素，并将后面的元素向前平移
                l -= 1
                i -= 1 #  让i停留在此处再次对比一下
            i += 1
        return l

class Solution1:
    def removeElement(self, nums: list[int], val: int) -> int:
        # 快慢指针
        fast = 0 # 快指针
        slow = 0 # 慢指针
        size = len(nums)
        while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
            if nums[fast] != val:
                # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow



