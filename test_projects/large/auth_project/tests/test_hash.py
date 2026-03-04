import unittest
from hash import simple_hash, hash_match, hash_salt, verify_hash, combine_hashes
class Test(unittest.TestCase):
    def test_hash(self): h=simple_hash("x"); self.assertTrue(hash_match("x", h))
    def test_salt(self): self.assertEqual(hash_salt("x","y"), simple_hash("xy"))
    def test_verify(self): self.assertTrue(verify_hash("x", simple_hash("x")))
    def test_combine(self): self.assertIsInstance(combine_hashes(1,2), int)