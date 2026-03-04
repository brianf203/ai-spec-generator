import unittest
from response import redirect_response, paginated_response, wrap_error, cors_headers
class TestEdge(unittest.TestCase):
    def test_redirect(self): r=redirect_response("/x"); self.assertEqual(r["status"], 302)
    def test_paginated(self): r=paginated_response([1,2], 1, 2, 10); self.assertEqual(r["total"], 10)
    def test_wrap(self): r=wrap_error(ValueError("x")); self.assertEqual(r["status"], 500)
    def test_cors(self): self.assertIn("Access-Control-Allow-Origin", cors_headers())