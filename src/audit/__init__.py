"""
Audit module for cryptographically hash-chained tamper-evident logging.

IMPORTANT TERMINOLOGY & ARCHITECTURAL DISCLOSURE:
- IMPLEMENTED: Cryptographically hash-chained tamper-evident audit ledger.
- NOT IMPLEMENTED: Decentralized blockchain consensus / peer-to-peer network.
"""

from .ledger import AuditBlock, TamperEvidentLedger

__all__ = ["AuditBlock", "TamperEvidentLedger"]
