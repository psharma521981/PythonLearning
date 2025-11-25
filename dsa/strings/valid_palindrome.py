class Solution:
    def isPalindrome(self, s: str) -> bool:
        if len(s) < 2:
            return False

        for i in range(len(s)):
            if s[i] != s[len(s)-i-1]:
                return False

        return True

test = Solution()
print(test.isPalindrome("ababa"))