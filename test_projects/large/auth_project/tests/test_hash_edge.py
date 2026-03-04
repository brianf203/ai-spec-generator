import unittest
from hash import simple_hash, hash_to_hex, hex_to_hash, constant_time_compare
class TestEdge(unittest.TestCase):
    def test_hash_int(self): self.assertIsInstance(simple_hash("x"), int)
    def test_hex_roundtrip(self): h=simple_hash("test"); self.assertEqual(hex_to_hash(hash_to_hex(h)), h % 1000)
    def test_constant_time_same(self): self.assertTrue(constant_time_compare("abc","abc"))
    def test_constant_time_diff_len(self): self.assertFalse(constant_time_compare("a","ab"))