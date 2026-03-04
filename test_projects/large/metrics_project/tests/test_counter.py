import unittest
from counter import increment, get_count, reset_counter, counter_keys
class TestCounter(unittest.TestCase):
    def test_increment(self): c={}; increment(c,"x"); self.assertEqual(get_count(c,"x"), 1)
    def test_reset(self): c={"x":5}; reset_counter(c,"x"); self.assertEqual(get_count(c,"x"), 0)
    def test_keys(self): c={}; increment(c,"a"); increment(c,"b"); self.assertEqual(set(counter_keys(c)), {"a","b"})