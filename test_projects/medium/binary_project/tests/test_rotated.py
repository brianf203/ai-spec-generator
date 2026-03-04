import unittest
from rotated import find_rotation_point, search_rotated, min_in_rotated
class Test(unittest.TestCase):
    def test_rot_point(self): self.assertEqual(find_rotation_point([4,5,1,2,3]), 2)
    def test_search(self): self.assertEqual(search_rotated([4,5,1,2,3], 2), 3)
    def test_min(self): self.assertEqual(min_in_rotated([4,5,1,2,3]), 1)