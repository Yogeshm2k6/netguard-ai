"""Central configuration for NetGuard AI."""

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length Fwd Packets", "Total Length Bwd Packets", "Fwd Packet Length Mean",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Fwd IAT Mean", "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
    "RST Flag Count", "FIN Flag Count", "Average Packet Size",
    "Init Win Bytes Forward", "Idle Mean",
]

CLASSES = ["Benign", "DoS/DDoS", "Probe", "Brute Force", "Web Attack", "Botnet", "Other"]

COLORS = {
    "Benign": "#22c55e", "DoS/DDoS": "#ef4444", "Probe": "#f59e0b",
    "Brute Force": "#a855f7", "Web Attack": "#3b82f6", "Botnet": "#eab308", "Other": "#94a3b8",
}

# Severity weight (0-100) used to turn class probabilities into a single Threat Score.
SEVERITY = {
    "Benign": 0, "Probe": 45, "Other": 60, "Brute Force": 70,
    "Web Attack": 75, "DoS/DDoS": 85, "Botnet": 92,
}

# Feature "families" used for grouped explanations.
GROUPS = {
    "Volume": ["Total Fwd Packets", "Total Backward Packets", "Total Length Fwd Packets",
               "Total Length Bwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
               "Average Packet Size"],
    "Timing / Rate": ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
                      "Fwd IAT Mean", "Idle Mean"],
    "TCP Flags": ["SYN Flag Count", "ACK Flag Count", "PSH Flag Count", "RST Flag Count",
                  "FIN Flag Count"],
    "Port & Window": ["Destination Port", "Init Win Bytes Forward"],
}

# Column aliases so real CICIDS2017 / CSE-CIC-IDS2018 CSVs work directly.
ALIASES = {
    "Total Length of Fwd Packets": "Total Length Fwd Packets",
    "Total Length of Bwd Packets": "Total Length Bwd Packets",
    "Init_Win_bytes_forward": "Init Win Bytes Forward",
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length Fwd Packets",
    "TotLen Bwd Pkts": "Total Length Bwd Packets",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Pkt Size Avg": "Average Packet Size",
    "Init Fwd Win Byts": "Init Win Bytes Forward",
}

PLAYBOOK = {
    "Benign": "No action required. Traffic matches normal behaviour; keep passive monitoring on.",
    "DoS/DDoS": "Enable rate-limiting / SYN cookies, push offending sources to a WAF or CDN scrubbing "
                "rule, and scale out the affected service. Preserve logs for post-incident review.",
    "Probe": "Reconnaissance detected. Block or tarpit the scanning source, close unused ports and "
             "review firewall exposure. Watch for follow-up exploitation from the same IP.",
    "Brute Force": "Enforce account lockout + MFA, deploy fail2ban-style blocking on SSH/FTP, and "
                   "rotate any credentials that may have been guessed.",
    "Web Attack": "Inspect the request payloads, enable/update WAF rules (SQLi/XSS), patch the web "
                  "application and review recent database and file-system activity.",
    "Botnet": "Likely C2 beaconing. Isolate the infected host, block the C2 destination at DNS/egress, "
              "run endpoint forensics and reset credentials used on that host.",
    "Other": "Unusual behaviour (possible infiltration / data exfiltration). Escalate to the SOC, "
             "isolate the host and capture full packet data for analysis.",
}


def map_label(raw) -> str:
    """Map a raw dataset label (CICIDS style) to one of our 7 classes."""
    s = str(raw).strip().lower()
    if s in ("benign", "normal"):
        return "Benign"
    if "dos" in s:
        return "DoS/DDoS"
    if "scan" in s or "probe" in s:
        return "Probe"
    if "patator" in s or "brute" in s:
        return "Brute Force"
    if "web attack" in s or "sql" in s or "xss" in s:
        return "Web Attack"
    if "bot" in s:
        return "Botnet"
    return "Other"


def threat_level(score: float):
    if score < 15:
        return "LOW", "#22c55e"
    if score < 45:
        return "MEDIUM", "#f59e0b"
    if score < 75:
        return "HIGH", "#f97316"
    return "CRITICAL", "#ef4444"
