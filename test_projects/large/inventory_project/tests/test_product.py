import unittest
from product import create_product, get_product_name, get_product_sku, get_product_price, validate_product, product_eq
class Test(unittest.TestCase):
    def test_create(self): p=create_product("x","s1",9.99); self.assertEqual(p["name"],"x")
    def test_neg(self): self.assertRaises(ValueError, lambda: create_product("x","s",-1))
    def test_get(self): self.assertEqual(get_product_name({"name":"a"}), "a")
    def test_sku(self): self.assertEqual(get_product_sku({"sku":"S1"}), "S1")
    def test_validate(self): self.assertTrue(validate_product({"name":"a","sku":"s","price":1}))
    def test_eq(self): self.assertTrue(product_eq({"sku":"a"},{"sku":"a"}))