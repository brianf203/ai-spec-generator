import unittest
from product import create_product, get_product_price, validate_product, product_repr
class TestEdge(unittest.TestCase):
    def test_price_default(self): self.assertEqual(get_product_price({}), 0)
    def test_validate_missing(self): self.assertFalse(validate_product({"name":"x"}))
    def test_repr(self): self.assertIn("Product", product_repr({"name":"x","sku":"s"}))
    def test_update(self): p=create_product("a","s",1); p["price"]=2; self.assertEqual(p["price"], 2)