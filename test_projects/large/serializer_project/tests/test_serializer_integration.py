import unittest
from json_util import to_json, from_json
from pickle_util import to_pickle, from_pickle
class TestIntegration(unittest.TestCase):
    def test_json_pickle_both(self): d={"a":1}; self.assertEqual(from_json(to_json(d)), from_pickle(to_pickle(d)))
    def test_nested_roundtrip(self): d={"a":[1,2]}; self.assertEqual(from_json(to_json(d)), d)