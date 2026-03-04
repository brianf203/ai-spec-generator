import unittest
from template import render_template, validate_template, get_placeholders
class TestTemplate(unittest.TestCase):
    def test_render(self): self.assertEqual(render_template("Hi {{name}}", {"name":"Bob"}), "Hi Bob")
    def test_validate(self): self.assertTrue(validate_template("{{a}}{{b}}", ["a","b"]))
    def test_placeholders(self): self.assertEqual(get_placeholders("{{x}} and {{y}}"), ["x","y"])