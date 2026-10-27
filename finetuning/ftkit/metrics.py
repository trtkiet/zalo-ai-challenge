"""Dependency-free corpus error rates (counts are summed before division)."""
from .common import normalize


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def error_rates(pairs):
    word_errors = words = char_errors = chars = 0
    for reference, hypothesis in pairs:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
        word_errors += edit_distance(reference.split(), hypothesis.split())
        words += len(reference.split())
        reference, hypothesis = reference.replace(" ", ""), hypothesis.replace(" ", "")
        char_errors += edit_distance(reference, hypothesis)
        chars += len(reference)
    return {"wer": word_errors / words if words else None, "cer": char_errors / chars if chars else None,
            "word_errors": word_errors, "reference_words": words, "char_errors": char_errors, "reference_chars": chars}
