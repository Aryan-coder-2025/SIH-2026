"""
Tamper-Evident Hash-Chained Audit Ledger for SIH-26153.

ARCHITECTURAL DISCLOSURE & SYSTEM BOUNDARIES:
- IMPLEMENTED: Cryptographically hash-chained tamper-evident audit ledger using SHA-256
  for immutable recording of multi-horizon attack forecasts, MITRE ATT&CK evidence,
  and SOC recommendations.
- NOT IMPLEMENTED: Decentralized blockchain consensus, peer-to-peer gossip protocol,
  or distributed proof-of-work/stake networks.

This implementation guarantees:
1. Append-only integrity verification via SHA-256 block hash chaining.
2. Tamper detection for:
   - Modified payload data
   - Modified previous_hash pointers
   - Reordered blocks
   - Deleted blocks
   - Duplicated blocks
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AuditBlock:
    """
    Individual audit block in the hash-chained ledger.
    """
    index: int
    timestamp: float
    record_id: str
    payload: Dict[str, Any]
    previous_hash: str
    block_hash: str

    @staticmethod
    def compute_hash(index: int, timestamp: float, record_id: str, payload: Dict[str, Any], previous_hash: str) -> str:
        """
        Compute deterministic SHA-256 hash across canonical serialized fields.
        """
        content = {
            "index": index,
            "timestamp": round(float(timestamp), 6),
            "record_id": str(record_id),
            "payload": payload,
            "previous_hash": str(previous_hash)
        }
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Check if block_hash matches recomputed hash."""
        recomputed = self.compute_hash(
            index=self.index,
            timestamp=self.timestamp,
            record_id=self.record_id,
            payload=self.payload,
            previous_hash=self.previous_hash
        )
        return self.block_hash == recomputed


class TamperEvidentLedger:
    """
    A cryptographically hash-chained tamper-evident audit ledger.
    """

    GENESIS_PREVIOUS_HASH = "0" * 64

    def __init__(self, chain: Optional[List[AuditBlock]] = None):
        self.chain: List[AuditBlock] = []
        if chain:
            self.chain = list(chain)

    def append_record(
        self,
        record_id: str,
        payload: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> AuditBlock:
        """
        Append a new verifiable audit record to the ledger.
        """
        ts = time.time() if timestamp is None else float(timestamp)
        index = len(self.chain)

        if index == 0:
            previous_hash = self.GENESIS_PREVIOUS_HASH
        else:
            previous_hash = self.chain[-1].block_hash

        block_hash = AuditBlock.compute_hash(
            index=index,
            timestamp=ts,
            record_id=record_id,
            payload=payload,
            previous_hash=previous_hash
        )

        block = AuditBlock(
            index=index,
            timestamp=ts,
            record_id=record_id,
            payload=payload,
            previous_hash=previous_hash,
            block_hash=block_hash
        )
        self.chain.append(block)
        return block

    def verify_ledger_integrity(self) -> Tuple[bool, List[str]]:
        """
        Comprehensive audit verification of the entire ledger chain.
        Returns:
            (is_valid: bool, violations: List[str])
        """
        violations: List[str] = []

        if not self.chain:
            return True, violations

        for i, block in enumerate(self.chain):
            # 1. Index verification (detects reordering, deletion, duplication)
            if block.index != i:
                violations.append(
                    f"Block at position {i} has invalid index {block.index} (expected {i})"
                )

            # 2. Previous hash linkage verification
            if i == 0:
                if block.previous_hash != self.GENESIS_PREVIOUS_HASH:
                    violations.append(
                        f"Genesis block has invalid previous_hash: {block.previous_hash}"
                    )
            else:
                expected_prev = self.chain[i - 1].block_hash
                if block.previous_hash != expected_prev:
                    violations.append(
                        f"Block {i} previous_hash mismatch: expected {expected_prev}, got {block.previous_hash}"
                    )

            # 3. Block content integrity / cryptographic hash verification
            if not block.verify_hash():
                violations.append(
                    f"Block {i} payload or metadata tampered: hash mismatch. "
                    f"Stored {block.block_hash}, recomputed "
                    f"{AuditBlock.compute_hash(block.index, block.timestamp, block.record_id, block.payload, block.previous_hash)}"
                )

            # 4. Monotonic timestamp verification
            if i > 0 and block.timestamp < self.chain[i - 1].timestamp:
                violations.append(
                    f"Block {i} timestamp ({block.timestamp}) earlier than preceding block ({self.chain[i - 1].timestamp})"
                )

        return (len(violations) == 0), violations

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ledger to dictionary."""
        return {
            "ledger_type": "tamper_evident_hash_chained_audit_ledger",
            "blockchain_consensus_implemented": False,
            "block_count": len(self.chain),
            "chain": [asdict(b) for b in self.chain]
        }

    def save_to_file(self, filepath: str) -> None:
        """Save ledger to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "TamperEvidentLedger":
        """Load ledger from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        blocks = [
            AuditBlock(
                index=b["index"],
                timestamp=b["timestamp"],
                record_id=b["record_id"],
                payload=b["payload"],
                previous_hash=b["previous_hash"],
                block_hash=b["block_hash"]
            )
            for b in data.get("chain", [])
        ]
        return cls(chain=blocks)
