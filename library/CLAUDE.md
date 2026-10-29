# Volt Library

A free, public, searchable directory of resources for people working where law meets technology: cybersecurity, data privacy and AI. Visitors browse and filter; anyone should be able to suggest a resource, and a curator reviews suggestions before they go live.

## Current state

- `index.html` is the site (plain HTML/CSS/JS, no build step). It fetches `data/resources.json`, so serve the folder (`python3 -m http.server`) rather than opening the file directly.
- `data/catalogue.xlsx` is the source of truth: 353 resources in 13 sections, researched and link-checked as of Oct 29, 2026. Its Read Me tab documents every column; Coverage and Removed tabs are for the curator.
- `scripts/import_catalogue.py` (needs openpyxl) validates the spreadsheet and writes `data/resources.json`. Never edit the JSON by hand; edit the spreadsheet and re-run the import (`--dry-run` first).
- Topics: Cybersecurity (c), Privacy (p), AI (a), Tech Law (l), General (g). Each has a `--t-*` colour token in light and dark mode.
- Search, topic filter, section navigation (URL hash, e.g. `#laws`) and auto-hiding of past events all work.
- The suggestion form and curator review queue were built for a Claude artifact's `window.claude` database. Outside that environment `window.claude` is undefined, so the form falls back to "copy your suggestion". It needs a real backend.

## History

- The first version came from the owner's Notion workspace (database "Volt Library") through a Notion sync script. The catalogue spreadsheet replaced it on Oct 29, 2026: 42 Notion entries were kept (some renamed or updated to their next edition) and the rest are listed with reasons on the spreadsheet's Removed tab. The Notion sync was removed so it can't overwrite the catalogue; it is in git history (commit aaf00a9) if needed.
- The Notion page still carries the original template creator's social links and disclaimer, if it is kept public.

## Known content gaps

- 2 links are marked "Needs check" (UNCTAD privacy legislation tracker, Euractiv). Curators see a "Check link" pill on them.
- Global Privacy Assembly 2027 has no dates yet.

## Suggested next steps

1. Give suggestions a real home: a form service (the owner already uses Typeform) or a small backend with a review queue.
2. Deploy as a static site (e.g. GitHub Pages, Netlify or Vercel).
3. Consider filters for level and cost now that the data has them.

## Design

Volt LegalTech Lab identity: black #0A0A0A, white, red #E11D2E and blue #2563EB, with the short red-and-blue stripe as a recurring rule. Fonts: Newsreader (display), IBM Plex Sans (body), IBM Plex Mono (call numbers and labels). No logo on the site; the header is the plain site name. Light and dark themes via `prefers-color-scheme`.

The interface is a virtual library: each section is a bookcase, each resource a spine coloured by its first topic (Cybersecurity red, Privacy blue, AI dark grey, Tech Law black in light mode and white in dark mode, General light grey) with bands for further topics and a call-number label (the catalogue ID). Clicking a spine opens its catalogue card. The Shelves/List toggle keeps a dense list view for scanning; the choice is remembered per browser.
