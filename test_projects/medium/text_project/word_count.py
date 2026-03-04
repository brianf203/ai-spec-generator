"""Word count."""
def word_count(s):
    """Count words in s. Returns 0 for empty/whitespace-only strings."""
    if not s or not s.strip():
        return 0
    return len(s.split())

def sentence_count(s):
    """Count sentences by counting sentence-ending punctuation."""
    count = s.count(".") + s.count("!") + s.count("?")
    return count

def line_count(s):
    """Count lines in s."""
    return len(s.splitlines())

def paragraph_count(s):
    """Count non-empty paragraphs (blocks separated by double newline)."""
    paragraphs = s.split("\n\n")
    count = 0
    for p in paragraphs:
        if p.strip():
            count += 1
    return count
def avg_word_length(s):
    words = s.split()
    if not words:
        return 0
    total_len = 0
    for w in words:
        total_len += len(w)
    return total_len / len(words)
def max_word_length(s):
    """Return length of longest word. Returns 0 if no words."""
    words = s.split()
    if not words:
        return 0
    max_len = len(words[0])
    for w in words[1:]:
        if len(w) > max_len:
            max_len = len(w)
    return max_len
def word_frequency(s):
    from collections import Counter
    return dict(Counter(s.lower().split()))
def unique_words(s):
    """Return set of unique words (lowercased)."""
    result = set()
    for w in s.split():
        result.add(w.lower())
    return result