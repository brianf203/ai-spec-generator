import unittest
from utils import unique_list, set_from_iterable, frozen_from, set_size, set_contains
class TestUtils(unittest.TestCase):
    def test_unique(self): self.assertEqual(sorted(unique_list([1,2,1])), [1,2])
    def test_set_from(self): self.assertEqual(set_from_iterable([1,2]), {1,2})
    def test_frozen(self): self.assertEqual(frozen_from([1,2]), frozenset({1,2}))
    def test_size(self): self.assertEqual(set_size({1,2,3}), 3)
    def test_contains(self): self.assertTrue(set_contains({1,2}, 1))