"""Release restoration of the voting helper absent from the source archive.

The rule is taken from thr_label.py's module docstring and statistics output.
This is not a recovered copy of the historical offline relabeling CLI.
"""


def final_label(labels):
    """2 abstains; label 1 requires at least one 1 and no more than one 0.

    The caller drops the all-2 case before aggregation. The standalone helper
    returns 0 for that case, matching the documented 'otherwise 0' rule.
    """
    labels = tuple(labels)
    if len(labels) != 3 or any(type(v) is not int or v not in (0, 1, 2) for v in labels):
        raise ValueError("Expected exactly three integer labels in {0, 1, 2}")
    return int(labels.count(0) <= 1 and labels.count(1) > 0)
