"""
Protocol Normalization Module.

Defines the canonical numeric protocol representation across the SIH pipeline:
    TCP   = 6
    UDP   = 17
    ICMP  = 1
    OTHER = -1
"""
from typing import Any

PROTOCOL_TCP = 6
PROTOCOL_UDP = 17
PROTOCOL_ICMP = 1
PROTOCOL_OTHER = -1

CANONICAL_PROTOCOL_MAP: dict[str, int] = {
    "TCP": PROTOCOL_TCP,
    "6": PROTOCOL_TCP,
    "UDP": PROTOCOL_UDP,
    "17": PROTOCOL_UDP,
    "ICMP": PROTOCOL_ICMP,
    "1": PROTOCOL_ICMP,
}


def normalize_protocol(value: Any) -> int:
    """
    Convert protocol representation into canonical numeric IDs.

    Canonical Mapping:
        TCP   -> 6
        UDP   -> 17
        ICMP  -> 1
        OTHER -> -1

    Handles:
        - String names: 'TCP', 'udp', 'ICMP'
        - Numeric values: 6, 17, 1, 6.0, '6', '17', '1'
        - Unknown / non-canonical protocols (e.g. 47 / GRE, 2 / IGMP, 'ARP') -> -1
        - None, empty string, NaN, missing values -> -1
        - Scapy packet objects containing TCP, UDP, ICMP, or IP layers.
    """
    if value is None:
        return PROTOCOL_OTHER

    # Handle Scapy packet objects dynamically if passed
    if hasattr(value, "haslayer"):
        try:
            from scapy.all import IP, TCP, UDP, ICMP  # type: ignore

            if value.haslayer(TCP):
                return PROTOCOL_TCP
            if value.haslayer(UDP):
                return PROTOCOL_UDP
            if value.haslayer(ICMP):
                return PROTOCOL_ICMP
            if value.haslayer(IP):
                proto = int(value[IP].proto)
                if proto == PROTOCOL_TCP:
                    return PROTOCOL_TCP
                if proto == PROTOCOL_UDP:
                    return PROTOCOL_UDP
                if proto == PROTOCOL_ICMP:
                    return PROTOCOL_ICMP
            return PROTOCOL_OTHER
        except Exception:
            return PROTOCOL_OTHER

    # Handle numeric values (int, float, numpy types)
    if isinstance(value, (int, float)):
        try:
            import math
            if not math.isfinite(value):
                return PROTOCOL_OTHER
            val_int = int(value)
            if val_int in (PROTOCOL_TCP, PROTOCOL_UDP, PROTOCOL_ICMP):
                return val_int
            return PROTOCOL_OTHER
        except (ValueError, OverflowError):
            return PROTOCOL_OTHER

    # Handle string values
    str_val = str(value).strip().upper()
    if not str_val or str_val == "NAN" or str_val == "NONE":
        return PROTOCOL_OTHER

    # Handle numeric float strings like "6.0"
    try:
        f_val = float(str_val)
        import math
        if math.isfinite(f_val) and f_val.is_integer():
            int_val = int(f_val)
            if int_val in (PROTOCOL_TCP, PROTOCOL_UDP, PROTOCOL_ICMP):
                return int_val
            return PROTOCOL_OTHER
    except ValueError:
        pass

    return CANONICAL_PROTOCOL_MAP.get(str_val, PROTOCOL_OTHER)