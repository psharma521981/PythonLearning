from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        aMap = {}
        for i in range(len(nums)):
            if (target - nums[i]) in aMap:
                return [aMap.get(target - nums[i]), i]
            aMap[nums[i]] = i
        return []

test = Solution()
print(test.twoSum([2,3,4,5,6], 9))
