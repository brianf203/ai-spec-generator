import unittest
from digits import sum_digits, product_digits, digit_count, reverse_digits, is_palindrome_number, digit_root, contains_digit
class Test(unittest.TestCase):
    def test_sum(self): self.assertEqual(sum_digits(123), 6)
    def test_product(self): self.assertEqual(product_digits(123), 6)
    def test_count(self): self.assertEqual(digit_count(12345), 5)
    def test_reverse(self): self.assertEqual(reverse_digits(123), 321)
    def test_palindrome(self): self.assertTrue(is_palindrome_number(121))
    def test_root(self): self.assertEqual(digit_root(38), 2)
    def test_contains(self): self.assertTrue(contains_digit(123, 2))