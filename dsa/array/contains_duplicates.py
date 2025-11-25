from typing import List


class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

test = Solution()
print(test.hasDuplicate([1,1,3,4,5,26]))
