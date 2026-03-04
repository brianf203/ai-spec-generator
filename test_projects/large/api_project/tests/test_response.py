import unittest
from response import json_response, error_response, success_response, created_response, no_content_response
class Test(unittest.TestCase):
    def test_json(self): r=json_response({"x":1}); self.assertEqual(r["status"], 200)
    def test_err(self): r=error_response(400,"bad"); self.assertEqual(r["status"], 400)
    def test_success(self): r=success_response({"a":1}); self.assertEqual(r["status"], 200)
    def test_created(self): r=created_response({"id":1}); self.assertEqual(r["status"], 201)
    def test_no_content(self): self.assertEqual(no_content_response()["status"], 204)