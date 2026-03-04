import unittest
from window import window_key, is_in_window, next_window_start
class TestWindow(unittest.TestCase):
    def test_window_key(self): self.assertEqual(window_key(100, 60), 1)
    def test_in_window(self): self.assertTrue(is_in_window(50, 0, 60))
    def test_next(self): self.assertEqual(next_window_start(50, 60), 60)