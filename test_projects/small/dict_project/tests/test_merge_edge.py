import unittest
from merge import merge_dicts, invert_dict, merge_three
class TestMergeEdge(unittest.TestCase):
    def test_merge_empty(self): self.assertEqual(merge_dicts({}, {"a":1}), {"a":1})
    def test_merge_override(self): self.assertEqual(merge_dicts({"a":1},{"a":2}), {"a":2})
    def test_invert_dup(self): d=invert_dict({"a":1,"b":1}); self.assertIn(d[1], ("a", "b"))
    def test_merge_three_empty(self): self.assertEqual(merge_three({},{},{}), {})