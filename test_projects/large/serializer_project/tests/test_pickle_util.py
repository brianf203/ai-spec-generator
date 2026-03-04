import unittest
from pickle_util import to_pickle, from_pickle, pickle_copy, pickle_roundtrip, to_base64_pickle, from_base64_pickle
class Test(unittest.TestCase):
    def test_roundtrip(self): self.assertEqual(from_pickle(to_pickle([1,2])), [1,2])
    def test_copy(self): self.assertEqual(pickle_copy([1,2]), [1,2])
    def test_roundtrip2(self): self.assertEqual(pickle_roundtrip({"a":1}), {"a":1})
    def test_base64(self): self.assertEqual(from_base64_pickle(to_base64_pickle(42)), 42)