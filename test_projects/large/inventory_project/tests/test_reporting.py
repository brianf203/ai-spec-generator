import unittest
from reporting import total_value, low_stock_threshold, inventory_value, low_stock_report, fill_rate
class Test(unittest.TestCase):
    def test_value(self): self.assertEqual(total_value(10,5), 50)
    def test_low(self): self.assertTrue(low_stock_threshold(5,10))
    def test_inv_value(self): self.assertEqual(inventory_value([{"qty":2,"price":5}]), 10)
    def test_low_report(self): self.assertEqual(len(low_stock_report([{"qty":1},{"qty":10}], 5)), 1)
    def test_fill(self): self.assertEqual(fill_rate(8, 10), 0.8)