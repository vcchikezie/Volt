#!/usr/bin/env python3
"""Turn the Volt Library catalogue spreadsheet into data/resources.json.

    python3 scripts/import_catalogue.py              # reads data/catalogue.xlsx
    python3 scripts/import_catalogue.py --dry-run    # report only, write nothing

Needs openpyxl (pip install openpyxl). The spreadsheet is the source of truth;
edit it, not resources.json. Its Read Me tab explains each column.
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

try:
    import openpyxl
except ImportError:
    sys.exit("This script needs openpyxl. Install it with: pip install openpyxl")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = ROOT / "data" / "catalogue.xlsx"
OUT = ROOT / "data" / "resources.json"

TOPICS = {"c": "Cybersecurity", "p": "Privacy", "a": "AI", "l": "Tech Law", "g": "General"}
TOPIC_CODE = {v: k for k, v in TOPICS.items()}
# Spellings that mean the same place, so the site shows one label.
JURISDICTION_ALIASES = {"United States": "US", "European Union": "EU"}
REQUIRED = ["ID", "Name", "Section Key", "Type", "Topics", "Description", "URL"]


def text(v):
    return str(v).strip() if v is not None else ""


def iso(v):
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    s = text(v)
    if s:
        datetime.strptime(s, "%Y-%m-%d")  # raises on anything that isn't a real date
    return s


def format_when(start, end):
    """'Nov 2, 2026', 'Nov 2–4, 2026', 'Nov 30 – Dec 2, 2026' or 'Dec 30, 2026 – Jan 2, 2027'."""
    if not start:
        return ""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d") if end else s
    md = lambda d: f"{d:%b} {d.day}"
    if s == e:
        return f"{md(s)}, {s.year}"
    if s.year != e.year:
        return f"{md(s)}, {s.year} – {md(e)}, {e.year}"
    if s.month != e.month:
        return f"{md(s)} – {md(e)}, {e.year}"
    return f"{md(s)}–{e.day}, {e.year}"


def rows(ws):
    it = ws.iter_rows(values_only=True)
    header = [text(h) for h in next(it)]
    for n, r in enumerate(it, start=2):
        if any(v not in (None, "") for v in r):
            yield n, dict(zip(header, r))


def build(xlsx):
    wb = openpyxl.load_workbook(xlsx)
    errors, warnings = [], []

    sections = []
    for n, r in rows(wb["Sections"]):
        key = text(r.get("Section Key"))
        if not key:
            continue  # the Total row
        sections.append({"key": key, "code": text(r["Code"]), "name": text(r["Display Name"]), "description": text(r["Description"])})
    section_keys = {s["key"] for s in sections}

    resources, ids, names, checked = [], set(), set(), []
    for n, r in rows(wb["Resources"]):
        where = f"Resources row {n} ({text(r.get('ID')) or text(r.get('Name')) or 'no name'})"
        missing = [c for c in REQUIRED if not text(r.get(c))]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
            continue

        rid, name, section = text(r["ID"]), text(r["Name"]), text(r["Section Key"])
        if rid in ids:
            errors.append(f"{where}: ID {rid} is used twice")
        if name.lower() in names:
            warnings.append(f"{where}: another row has the same name")
        ids.add(rid)
        names.add(name.lower())
        if section not in section_keys:
            errors.append(f"{where}: section key '{section}' isn't on the Sections tab")

        topics = []
        for t in text(r["Topics"]).split(";"):
            t = t.strip()
            if not t:
                continue
            if t not in TOPIC_CODE:
                errors.append(f"{where}: unknown topic '{t}' (use {'; '.join(TOPICS.values())})")
            elif TOPIC_CODE[t] not in topics:
                topics.append(TOPIC_CODE[t])

        url = text(r["URL"])
        if not url.startswith(("https://", "http://")):
            errors.append(f"{where}: URL doesn't start with https://")
        desc = text(r["Description"])
        if len(desc) > 160:
            warnings.append(f"{where}: description is {len(desc)} characters (limit 160)")

        try:
            start, end = iso(r.get("Start Date")), iso(r.get("End Date"))
        except ValueError:
            errors.append(f"{where}: Start/End Date must be a real date (YYYY-MM-DD)")
            start = end = ""
        if start and end and end < start:
            errors.append(f"{where}: End Date is before Start Date")

        status = text(r.get("Status"))
        jurisdiction = text(r.get("Jurisdiction"))
        item = {
            "id": rid,
            "name": name,
            "section": section,
            "kind": text(r["Type"]),
            "topics": topics,
            "description": desc,
            "url": url,
            "publisher": text(r.get("Publisher")),
            "jurisdiction": JURISDICTION_ALIASES.get(jurisdiction, jurisdiction),
            "level": text(r.get("Level")),
            "cost": text(r.get("Cost")),
            "format": text(r.get("Format")),
            "status": status,
        }
        if section == "events":
            item["event"] = {
                "when": format_when(start, end),
                "end": end or start,
                "location": text(r.get("Location")),
            }
            if item["event"]["end"] and item["event"]["end"] < date.today().isoformat():
                warnings.append(f"{where}: event has ended; it will be hidden as past")
        elif start:
            item["applies"] = start
        if text(r.get("Link Check")) == "Needs check":
            item["needsCheck"] = True
        if isinstance(r.get("Last Checked"), (datetime, date)):
            checked.append(iso(r["Last Checked"]))

        resources.append(prune(item))

    data = {
        "updated": max(checked) if checked else date.today().isoformat(),
        "topics": TOPICS,
        "sections": sections,
        "resources": resources,
    }
    return data, errors, warnings


def prune(d):
    return {k: prune(v) if isinstance(v, dict) else v for k, v in d.items() if v not in ("", None, [], {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xlsx", nargs="?", default=DEFAULT_XLSX, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="write even if the library would shrink by more than half")
    args = ap.parse_args()

    data, errors, warnings = build(args.xlsx)
    for w in warnings:
        print(f"! {w}")
    if errors:
        for e in errors:
            print(f"x {e}", file=sys.stderr)
        sys.exit(f"\n{len(errors)} problem(s) in {args.xlsx.name}. Nothing written.")

    before = json.loads(OUT.read_text()) if OUT.exists() else {"resources": []}
    old = {r["name"]: r for r in before["resources"]}
    new = {r["name"]: r for r in data["resources"]}
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    changed = [n for n in new if n in old and new[n] != old[n]]

    print(f"\n{len(data['resources'])} resources in {len(data['sections'])} sections")
    for label, names in (("Added", added), ("Removed", removed), ("Changed", changed)):
        if names:
            print(f"{label}: {len(names)}")

    if len(new) < len(old) / 2 and not args.force:
        sys.exit(f"Refusing to write: resources would drop from {len(old)} to {len(new)}. Re-run with --force if that's intended.")
    if args.dry_run:
        print("Dry run, nothing written.")
        return
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
