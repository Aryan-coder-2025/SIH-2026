# Data Join Contract

## Purpose

This document defines how packet-level and flow-level features are joined
into the canonical network-state representation.

The join must use the same identity and time definitions across the entire
SIH pipeline.

## 1. Primary Identity

The primary identity is:

- `src_ip`

This represents the source host generating the network traffic.

## 2. Temporal Identity

The temporal identity is:

- `window_id`

Each `window_id` represents one canonical 10-second time window.

Required time fields:

- `timestamp` — original packet/flow timestamp
- `window_start` — beginning of the 10-second window
- `window_end` — end of the 10-second window
- `window_id` — integer identifying the 10-second window

## 3. Optional Flow Identity

When flow-level matching is required, use:

`src_ip + dst_ip + src_port + dst_port + protocol`

## 4. Protocol Representation

Protocol must use the canonical numeric protocol ID.

| Protocol | ID |
|---|---:|
| TCP | 6 |
| UDP | 17 |
| ICMP | 1 |
| OTHER | -1 |

Examples:

- `TCP` → `6`
- `"tcp"` → `6`
- `"6"` → `6`
- `UDP` → `17`
- `ICMP` → `1`
- Unknown protocol → `-1`

## 5. Network State

One state represents:

`one source host × one 10-second time window`

Therefore:

- One row is NOT one packet.
- One row is NOT one flow.
- One row represents the aggregated state of a source host during a
  10-second window.

Example:

`src_ip = 192.168.10.15`
`window_id = 250`

This represents one network-state vector.

## 6. Packet Feature Input

Packet-level features include:

- `packet_count`
- `ttl_mean`
- `ttl_std`
- `ttl_min`
- `ttl_max`
- `tcp_window_mean`
- `tcp_window_std`
- `fragment_count`
- `payload_mean`
- `payload_std`
- `payload_min`
- `payload_max`
- `retransmission_count`
- `port_scan_score`
- `sequential_port_ratio`
- `unique_dst_ports`
- `packet_iat_mean`
- `packet_iat_std`
- `packet_iat_max`

## 7. Join Validation

Every flow + packet join must report:

- number of flow rows before join
- number of packet rows before join
- number of rows after join
- packet coverage percentage
- flow coverage percentage
- unmatched packet windows
- unmatched flow windows

The join must be validated before the combined feature matrix is passed
to the forecasting/model pipeline.

## 8. Canonical Join Key

Primary join:

`src_ip + window_id`

Optional detailed flow identity:

`src_ip + dst_ip + src_port + dst_port + protocol`

## 9. Output

The final fused data must represent:

`source host × 10-second window`

and must be compatible with the project's temporal forecasting pipeline.