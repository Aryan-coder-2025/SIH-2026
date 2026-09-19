"""
Canonical Protocol Normalization Module.
Maps protocol representations to canonical numeric IDs according to project.yaml.
"""


def normalize_protocol(value):
    """
    Convert protocol values into canonical numeric IDs.

    TCP   = 6
    UDP   = 17
    ICMP  = 1
    OTHER = -1
    """
    if value is None:
        return -1

    # If already integer
    if isinstance(value, int):
        if value in (6, 17, 1):
            return value
        if value == 58:  # IPv6-ICMP
            return 1
        return -1

    val_str = str(value).strip().upper()

    protocol_map = {
        "TCP": 6,
        "6": 6,
        "UDP": 17,
        "17": 17,
        "ICMP": 1,
        "1": 1,
        "ICMPV6": 1,
        "ICMP6": 1,
        "58": 1,
    }

    return protocol_map.get(val_str, -1)