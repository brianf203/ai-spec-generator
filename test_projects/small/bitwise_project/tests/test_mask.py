import unittest
from mask import set_bit, clear_bit, toggle_bit, get_bit, mask_lower
class TestMask(unittest.TestCase):
    def test_set(self): self.assertEqual(set_bit(0, 2), 4)
    def test_clear(self): self.assertEqual(clear_bit(7, 1), 5)
    def test_toggle(self): self.assertEqual(toggle_bit(4, 2), 0)
    def test_get(self): self.assertEqual(get_bit(5, 0), 1)
    def test_mask(self): self.assertEqual(mask_lower(15, 2), 3)