def normalize_protocol(value):
    """
    Convert protocol values into canonical numeric IDs.

    TCP  = 6
    UDP  = 17
    ICMP = 1
    OTHER = -1
    """

    if value is None:
        return -1

    value = str(value).strip().upper()

    protocol_map = {
        "TCP": 6,
        "6": 6,
        "UDP": 17,
        "17": 17,
        "ICMP": 1,
        "1": 1,
    }

    return protocol_map.get(value, -1)