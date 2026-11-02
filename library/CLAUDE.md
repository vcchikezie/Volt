# Volt Library

Owned by Datavance Solutions LLC. A free, public, searchable directory of resources for people working where law meets technology: cybersecurity, data privacy and AI. Visitors browse and filter; anyone should be able to suggest a resource, and a curator reviews suggestions before they go live.

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

Volt Library belongs to Datavance Solutions LLC and follows the Datavance brand. The rules are in `brand/03-color.md` and `brand/04-typography.md`; read them before changing the look.

- Black and white carry the brand (Kuro #000, Shiro #FFF, and the board greys). Light is the default for this content site; dark mode follows `prefers-color-scheme` (black field, Graphite cards).
- Green #00A651, yellow #FFC20E and red #E0261F are signals only: a small dot (`.flag`) beside a word, where the state changes what the reader does (law in force / partly in force / not yet in force, dates to be announced, no longer updated, curator link checks). Never text, fills, headings or decoration. No blue, no purple, no gradients between colours.
- The one filled action per view is the Sente pill (`.btn.sente`): black on light, white on dark, JetBrains Mono capitals. Other actions are outline pills or links.
- Type: Newsreader Light 300 for statements at 32px and up (hero, section titles, card titles, cover monograms); Inter for everything read; JetBrains Mono for labels, buttons and identifiers (call numbers). Emphasis is Newsreader italic in ink, never colour.
- No logo on the site (the brand says the wordmark is not type, and the owner asked for no logo); the header reads "Volt" with a "by Datavance Solutions LLC" label. The footer credits Datavance Solutions LLC.
- Topic covers (the generated fallback thumbnails) use brand greys: Cybersecurity Kuro black, Privacy Slate Gray, AI Stone Gray, Tech Law Kaya Silver, General Shiro white. Topic dots keep the owner's colours, by explicit request and as an exception to the brand's no-blue rule: Cybersecurity red, Privacy blue, AI dark grey, Tech Law black (white in dark mode), General light grey.
- Thumbnails: `scripts/fetch_thumbnails.py` saves each resource's preview image (og:image/twitter:image, else its largest icon) to `data/thumbs/` and links it from `resources.json`; the importer keeps those links. It needs normal internet access (this repo's cloud sessions can't reach most resource sites). Tiles show the image, falling back to the generated cover when there is none or it fails to load.

The interface is a virtual library in a thumbnail view: each section shows its resources as tiles, each with a generated cover in its first topic's grey with a goban grid texture, a monogram (acronym or first letter), the call number (catalogue ID), and side bands for further topics; title, publisher, topics and any state flags sit under the cover. Clicking a tile opens its catalogue card. The Thumbnails/List toggle keeps a dense list view for scanning; the choice is remembered per browser. (The owner asked for a thumbnail grid like coloradovirtuallibrary.org, so the earlier "avoid card grids" rule no longer applies.)
