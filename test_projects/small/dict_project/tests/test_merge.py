import unittest
from merge import merge_dicts, invert_dict, merge_three
class TestMerge(unittest.TestCase):
    def test_merge(self): self.assertEqual(merge_dicts({"a":1},{"b":2}), {"a":1,"b":2})
    def test_invert(self): self.assertEqual(invert_dict({"a":1}), {1:"a"})
    def test_merge_three(self): self.assertEqual(merge_three({"a":1},{"b":2},{"c":3}), {"a":1,"b":2,"c":3})