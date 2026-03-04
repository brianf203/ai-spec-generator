"""Additional serializer tests."""
import unittest
from json_util import to_json, from_json
from pickle_util import to_pickle, from_pickle


class TestSerializerExtra(unittest.TestCase):
    def test_json_roundtrip(self):
        data = {"a": 1}
        s = to_json(data)
        self.assertEqual(from_json(s), data)

    def test_pickle_roundtrip(self):
        data = [1, 2, 3]
        b = to_pickle(data)
        self.assertEqual(from_pickle(b), data)
