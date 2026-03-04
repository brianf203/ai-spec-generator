import unittest
from stock import add_stock, remove_stock, set_stock, can_fulfill, stock_level, reorder_point
class Test(unittest.TestCase):
    def test_add(self): self.assertEqual(add_stock(10,5), 15)
    def test_remove(self): self.assertEqual(remove_stock(10,3), 7)
    def test_insufficient(self): self.assertRaises(ValueError, lambda: remove_stock(5, 10))
    def test_set(self): self.assertEqual(set_stock(10, 5), 5)
    def test_fulfill(self): self.assertTrue(can_fulfill(10, 5))
    def test_level(self): self.assertEqual(stock_level(5, 10, 20), "low")
    def test_reorder(self): self.assertTrue(reorder_point(5, 10))