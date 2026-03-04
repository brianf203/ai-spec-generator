import unittest
from stock import set_stock, adjust_stock, reserve_stock, release_reservation, stock_turnover, days_of_stock
class TestEdge(unittest.TestCase):
    def test_set_neg(self): self.assertEqual(set_stock(10, -5), 0)
    def test_adjust_neg(self): self.assertEqual(adjust_stock(5, -3), 2)
    def test_reserve_excess(self): self.assertEqual(reserve_stock(5, 10), 5)
    def test_release(self): self.assertEqual(release_reservation(5, 2), 7)
    def test_turnover_zero(self): self.assertEqual(stock_turnover(10, 0), 0)
    def test_days_zero_sales(self): self.assertEqual(days_of_stock(10, 0), 0)