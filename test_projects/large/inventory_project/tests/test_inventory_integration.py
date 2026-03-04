import unittest
from product import create_product, validate_product
from stock import add_stock, can_fulfill
from reporting import total_value, inventory_value
from utils import format_currency
class TestIntegration(unittest.TestCase):
    def test_product_stock_value(self): p=create_product("x","s",10); self.assertEqual(total_value(5, p["price"]), 50)
    def test_inv_value_list(self): self.assertEqual(inventory_value([{"qty":2,"price":5}]), 10)
    def test_fulfill_after_add(self): self.assertTrue(can_fulfill(add_stock(5, 5), 8))