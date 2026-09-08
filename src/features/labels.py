def classify_label(label):
    """
    Convert a raw CTU-13 label into a high-level traffic category.
    """

    label = label.replace("flow=", "").strip()

    if label.startswith("From-Botnet"):
        return "botnet"

    if label.startswith("To-Botnet"):
        return "normal"

    if label.startswith("From-Normal"):
        return "normal"

    if label.startswith("To-Normal"):
        return "normal"

    if label.startswith("Background"):
        return "background"

    if label.startswith("From-Background"):
        return "background"

    if label.startswith("To-Background"):
        return "background"

    return "unknown"

if __name__ == "__main__":
    examples = [
        "flow=Background",
        "flow=From-Normal-V42-Jist",
        "flow=To-Background-CVUT-Proxy",
    ]

    for label in examples:
        print(label, "->", classify_label(label))