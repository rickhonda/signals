#!/usr/bin/env python3
"""
Parse Windows Event Log XML records (as exported lines) into a flat event-table CSV.

Input:  windows-xml.log  (lines containing <Event ...>...</Event>, sometimes split across lines)
Output: dc_security.csv  (event table usable by signals.py)

This parser is intentionally "example-scope" (Windows EventLog XML specific).
Core stays domain-agnostic; this lives under examples/.
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}

import ipaddress

def normalize_ip(ip: str) -> str:
    if not ip or ip == "-":
        return ""
    try:
        addr = ipaddress.ip_address(ip)
        # IPv6-mapped IPv4 → return dotted quad
        if addr.version == 6 and addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(addr)
    except ValueError:
        return ip

def iter_event_xml_chunks(fp: Iterable[str]) -> Iterator[str]:
    """
    Yield complete <Event>...</Event> XML chunks.
    Handles both:
      - one-event-per-line
      - multi-line events (accumulate until </Event>)
    """
    buf: list[str] = []
    in_event = False

    for line in fp:
        line = line.strip()
        if not line:
            continue

        # Some files start mid-line; try to recover by finding "<Event"
        if "<Event" in line and not in_event:
            in_event = True
            # keep only from first "<Event"
            line = line[line.find("<Event") :]

        if in_event:
            buf.append(line)

            if "</Event>" in line:
                # keep only up to the closing tag (in case there is trailing junk)
                joined = " ".join(buf)
                end = joined.find("</Event>") + len("</Event>")
                yield joined[:end]
                buf = []
                in_event = False

    # If file ends mid-event, ignore partial buffer (or you could raise)


def _get_text(node: Optional[ET.Element]) -> Optional[str]:
    if node is None:
        return None
    txt = node.text
    return txt.strip() if txt is not None else None


def parse_one_event(xml_str: str) -> Optional[Dict[str, str]]:
    """
    Returns a flat dict for one event, or None if parsing fails.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    # System fields
    event_id = _get_text(root.find("e:System/e:EventID", NS))
    time_created = root.find("e:System/e:TimeCreated", NS)
    ts = time_created.attrib.get("SystemTime") if time_created is not None else None
    computer = _get_text(root.find("e:System/e:Computer", NS))
    channel = _get_text(root.find("e:System/e:Channel", NS))
    provider = root.find("e:System/e:Provider", NS)
    provider_name = provider.attrib.get("Name") if provider is not None else None

    # EventData fields (Data Name="...")
    data_map: Dict[str, str] = {}
    for d in root.findall("e:EventData/e:Data", NS):
        name = d.attrib.get("Name")
        val = _get_text(d)
        if name:
            data_map[name] = val if val is not None else ""

    # Normalize a few fields your channels.yaml references
    # channels.yaml uses: timestamp, event_id, user, src_ip, status
    out: Dict[str, str] = {
        "timestamp": ts or "",
        "event_id": event_id or "",
        "computer": computer or "",
        "channel": channel or "",
        "provider": provider_name or "",
        # common useful pivots
        "user": data_map.get("TargetUserName", ""),
        "domain": data_map.get("TargetDomainName", ""),
        "service": data_map.get("ServiceName", ""),
        "status": data_map.get("Status", ""),
        #"src_ip": data_map.get("IpAddress", ""),
        "src_ip": normalize_ip(data_map.get("IpAddress", "")),
        "src_port": data_map.get("IpPort", ""),
        # extra fields (harmless if unused)
        "ticket_options": data_map.get("TicketOptions", ""),
        "ticket_encryption_type": data_map.get("TicketEncryptionType", ""),
        "logon_guid": data_map.get("LogonGuid", ""),
        "transmitted_services": data_map.get("TransmittedServices", ""),
    }

    # Basic sanity: must have timestamp and event_id
    if not out["timestamp"] or not out["event_id"]:
        return None

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse Windows EventLog XML lines into a CSV event table.")
    ap.add_argument("--input", required=True, help="Input log file (e.g., windows-xml.log)")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., dc_security.csv)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)

    rows_written = 0
    rows_failed = 0

    fieldnames = [
        "timestamp",
        "event_id",
        "computer",
        "channel",
        "provider",
        "user",
        "domain",
        "service",
        "status",
        "src_ip",
        "src_port",
        "ticket_options",
        "ticket_encryption_type",
        "logon_guid",
        "transmitted_services",
    ]

    with in_path.open("r", encoding="utf-8", errors="replace") as f, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        w = csv.DictWriter(out_f, fieldnames=fieldnames)
        w.writeheader()

        for chunk in iter_event_xml_chunks(f):
            row = parse_one_event(chunk)
            if row is None:
                rows_failed += 1
                continue
            w.writerow(row)
            rows_written += 1

    print(f"Wrote {rows_written:,} rows to {out_path}")
    if rows_failed:
        print(f"Skipped {rows_failed:,} unparsable/partial events", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

