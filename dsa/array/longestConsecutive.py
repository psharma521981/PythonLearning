from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:

        if not nums:
            return 0

        aSet = set(nums)
        maxSeq = 1
        for i in nums:
            if (i - 1) not in aSet:
                num = i
                seq = 1
                while num + 1 in aSet:
                    num = num + 1
                    seq = seq + 1
                maxSeq = max(maxSeq, seq)
        return maxSeq
a =  Solution()
print(a.longestConsecutive([3,5,2,10,4,5]))
print(a.longestConsecutive([1,2,3,4,5,6,7,8]))