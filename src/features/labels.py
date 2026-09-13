#!/usr/bin/env python3
"""
labels.py
---------
Maps raw CTU-13 Label strings to high-level traffic categories.

Categories:
  - botnet   : flows that originate FROM a botnet host
  - normal   : legitimate user traffic
  - background: background/unknown traffic (excluded during training)
"""


def classify_label(label: str) -> str:
    """Return 'botnet', 'normal', or 'background' for a raw CTU-13 label.

    The function is tolerant of leading/trailing whitespace and the optional
    'flow=' prefix used in different versions of the CTU-13 dataset.
    """
    if not isinstance(label, str):
        return "background"

    label = label.strip().replace("flow=", "")

    if label.startswith("From-Botnet"):
        return "botnet"
    if label.startswith(("To-Botnet", "From-Normal", "To-Normal")):
        return "normal"
    # Our synthetic data uses plain 'botnet' / 'normal'
    if label.lower() == "botnet":
        return "botnet"
    if label.lower() == "normal":
        return "normal"
    # Background / unknown
    return "background"


if __name__ == "__main__":
    examples = [
        "flow=From-Botnet-V42-TCP",
        "flow=From-Normal-V42-Jist",
        "flow=Background",
        "flow=To-Background-CVUT-Proxy",
        "botnet",
        "normal",
        None,
    ]
    for ex in examples:
        print(f"{ex!r} -> {classify_label(ex)}")
