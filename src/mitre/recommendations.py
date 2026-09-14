"""
Defender Action Recommendations Engine.
Owner: Srijani (Cybersecurity + MITRE + Explainability)

This module generates concrete, actionable mitigation steps and SOC recommendations
based on the candidate MITRE ATT&CK technique and forecast urgency.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RecommendationAction:
    action_type: str  # Containment, Investigation, Remediation, Hardening
    priority: str     # Critical, High, Medium, Low
    title: str
    description: str
    command_example: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "command_example": self.command_example,
        }


RECOMMENDATION_CATALOG: Dict[str, List[RecommendationAction]] = {
    "T1046": [
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Investigate Source Host for Systematic Service Scanning",
            description="Audit host active processes and socket connections. Inspect whether host is executing nmap, masscan, or custom scanning scripts.",
            command_example="ss -tulpn | grep LISTEN ; ps aux | grep -E '(nmap|scan|nc)'",
        ),
        RecommendationAction(
            action_type="Containment",
            priority="High",
            title="Enforce Dynamic Rate Limiting & Firewall Ingress Filter",
            description="Temporarily rate-limit SYN packet burst from source IP at perimeter firewall or host iptables to neutralize reconnaissance.",
            command_example="iptables -A INPUT -p tcp -m tcp --tcp-flags SYN,ACK,FIN,RST SYN -m limit --limit 5/s --limit-burst 10 -j ACCEPT",
        ),
        RecommendationAction(
            action_type="Hardening",
            priority="Medium",
            title="Close Unused Perimeter Ports & Enable TCP SYN Cookies",
            description="Ensure non-essential listening services are disabled and TCP SYN cookies are enabled to prevent socket exhaustion.",
            command_example="sysctl -w net.ipv4.tcp_syncookies=1",
        ),
    ],
    "T1110": [
        RecommendationAction(
            action_type="Containment",
            priority="Critical",
            title="Enforce Account Lockout & Temporary IP Blacklisting",
            description="Throttle failed authentication attempts and dynamically block offending IP address on SSH/RDP/Web portals via Fail2ban.",
            command_example="fail2ban-client set sshd banip <SRC_IP>",
        ),
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Audit Authentication Logs for Targeted Usernames",
            description="Review auth.log or Event Viewer (Event ID 4625) to identify dictionary spray scope and compromised accounts.",
            command_example="grep 'Failed password' /var/log/auth.log | tail -n 50",
        ),
    ],
    "T1498": [
        RecommendationAction(
            action_type="Containment",
            priority="Critical",
            title="Engage Automated BGP Flowspec or Upstream Scrubbing",
            description="Redirect high-volume volumetric attack traffic to DDoS scrubbing center or apply source rate drops at upstream router.",
            command_example="iptables -I INPUT -p tcp --syn -m connlimit --connlimit-above 50 -j DROP",
        ),
        RecommendationAction(
            action_type="Remediation",
            priority="High",
            title="Activate Anti-DDoS Mitigation Profiles",
            description="Drop malformed/fragmented packets and tune TCP SYN backlog queue length on edge gateways.",
            command_example="sysctl -w net.ipv4.tcp_max_syn_backlog=4096",
        ),
    ],
    "T1499": [
        RecommendationAction(
            action_type="Containment",
            priority="High",
            title="Rate Limit Target Endpoint HTTP/App Concurrency",
            description="Enforce per-source connection limits and enable web application firewall layer 7 challenge response.",
            command_example="iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 30 -j REJECT",
        ),
        RecommendationAction(
            action_type="Remediation",
            priority="Medium",
            title="Adjust Application Server Worker Timeouts",
            description="Reduce keep-alive timeout and increase server connection backlog pool.",
            command_example="systemctl reload nginx",
        ),
    ],
    "T1071": [
        RecommendationAction(
            action_type="Containment",
            priority="Critical",
            title="Isolate Compromised Host & Terminate C2 Channel",
            description="Quarantine source endpoint from internal VLAN and sinkhole outbound C2 domain/IP at DNS resolver.",
            command_example="ip route add blackhole <DEST_IP>",
        ),
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Capture Full PCAP & Extract Payload Artifacts",
            description="Trigger forensic packet capture on interface to extract TLS certificates, HTTP headers, or binary staging payloads.",
            command_example="tcpdump -i eth0 host <SRC_IP> -w c2_forensics.pcap",
        ),
    ],
    "T1048": [
        RecommendationAction(
            action_type="Containment",
            priority="Critical",
            title="Sever Outbound Connections to Destination IP",
            description="Instantly terminate active established sessions from source host to remote destination IP to prevent data leakage.",
            command_example="conntrack -D -s <SRC_IP> -d <DEST_IP>",
        ),
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Audit File System Access & DLP Alerts",
            description="Identify confidential files accessed by source host processes in the last 15 minutes before outbound volume spike.",
            command_example="lsof -p <PID> ; auditctl -l",
        ),
    ],
    "T1190": [
        RecommendationAction(
            action_type="Containment",
            priority="Critical",
            title="Apply Emergency WAF Rule & Patch Vulnerable Service",
            description="Deploy virtual patch at WAF gateway and restart web daemon with security update.",
            command_example="modsecurity-crs-update",
        ),
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Inspect Web Server Access Logs for Exploit Signatures",
            description="Search HTTP access logs for SQLi, command injection, or path traversal patterns.",
            command_example="tail -n 100 /var/log/nginx/access.log | grep -E '(\\%27|\\.\\./|union|select)'",
        ),
    ],
    "UNMAPPED": [
        RecommendationAction(
            action_type="Investigation",
            priority="High",
            title="Perform General Host & Network Telemetry Audit",
            description="Forecast indicates elevated risk though specific MITRE technique signature is unmapped. Capture live network flows and inspect anomalous process trees.",
            command_example="tcpdump -i any -c 500 -nn ; top -b -n 1 | head -n 20",
        ),
        RecommendationAction(
            action_type="Containment",
            priority="Medium",
            title="Increase Telemetry Polling Rate & Monitor Host Alerts",
            description="Enable verbose packet tracing and alert SOC tier-2 analysts for continued temporal observation.",
            command_example="systemctl status auditd",
        ),
    ],
}


class RecommendationEngine:
    """
    Generates tailored defender recommendations for forecast results and candidate MITRE techniques.
    """

    def __init__(self, catalog: Optional[Dict[str, List[RecommendationAction]]] = None):
        self.catalog = catalog or RECOMMENDATION_CATALOG

    def get_recommendations(self, mitre_technique: Optional[str], risk_score: float = 0.5) -> List[Dict[str, Any]]:
        tech_key = mitre_technique if (mitre_technique and mitre_technique in self.catalog) else "UNMAPPED"
        actions = self.catalog.get(tech_key, self.catalog["UNMAPPED"])

        results = []
        for action in actions:
            priority = action.priority
            if risk_score < 0.50 and priority == "Critical":
                priority = "High"

            results.append({
                "action_type": action.action_type,
                "priority": priority,
                "title": action.title,
                "description": action.description,
                "command_example": action.command_example,
            })
        return results
