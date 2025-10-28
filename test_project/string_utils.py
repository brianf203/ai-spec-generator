"""
String utilities for testing
"""

def reverse_string(s):
    """Reverse a string"""
    return s[::-1]


def count_vowels(s):
    """Count vowels in a string"""
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)


def is_palindrome(s):
    """Check if a string is a palindrome"""
    s = s.lower().replace(' ', '')
    return s == s[::-1]


def capitalize_words(s):
    """Capitalize each word in a string"""
    return ' '.join(word.capitalize() for word in s.split())


def count_words(s):
    """Count words in a string"""
    return len(s.split())
