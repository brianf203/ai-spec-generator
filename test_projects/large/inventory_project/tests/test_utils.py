import unittest
from utils import format_currency, clamp, round_price, apply_discount, apply_tax, margin, validate_sku
class Test(unittest.TestCase):
    def test_curr(self): self.assertEqual(format_currency(10.5), "$10.50")
    def test_clamp(self): self.assertEqual(clamp(5,0,10), 5)
    def test_round(self): self.assertEqual(round_price(10.556), 10.56)
    def test_discount(self): self.assertEqual(apply_discount(100, 10), 90.0)
    def test_tax(self): self.assertEqual(apply_tax(100, 10), 110.0)
    def test_margin(self): self.assertEqual(margin(50, 100), 0.5)
    def test_validate_sku(self): self.assertTrue(validate_sku("ABC123"))