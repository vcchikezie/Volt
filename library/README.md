# Volt Library

This folder is the Volt Library website. All commands below run from inside `library/`.

A searchable directory of resources on cybersecurity, data privacy and AI.

## Run it locally

The page loads its content from `data/resources.json`, and browsers block that when you open `index.html` straight from disk. Serve the `library/` folder instead:

```bash
cd library
```

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000.

## Files

| File | What it is | Edit by hand? |
|---|---|---|
| `index.html` | The site | Yes |
| `data/resources.json` | What the site shows. Written by the sync script | No, it gets overwritten |
| `data/curation.json` | Sections, topics, rows to skip, and per-resource details Notion doesn't have | Yes |
| `scripts/sync-notion.mjs` | Pulls Notion into `data/resources.json` | |

## Syncing from Notion

### One-time setup

1. Go to https://www.notion.so/profile/integrations, create an **internal** integration and copy its secret.
2. In Notion, open the **Resources** database, choose **•••** → **Connections**, and add the integration. Do the same for **Resource Categories**.

### Each sync

You need Node.js 18 or newer. There is nothing to install.

```bash
NOTION_TOKEN=your_secret node scripts/sync-notion.mjs --dry-run
```

This reports what would change without writing anything. If it looks right, run it again without `--dry-run`. Commit `data/resources.json` afterwards.

### How Notion and curation.json combine

- **Notion wins.** For any field filled in Notion (link, tags, category, dates, location, cost, description), the Notion value is used. Fixing something in Notion is enough.
- **curation.json fills gaps.** When the Notion field is empty, the value in `curation.json` is used, matched by resource name. This holds the links, sections and topics added during the first import, plus the "kind" label (e.g. "Job board"), which Notion has no field for.
- **New Notion rows** appear automatically if they have a Category. Tags map to topics; with no tags a row is filed under General.
- **Sections** are matched by Notion category name using each section's `notion` list in `curation.json`. If your Notion category is named differently (say "Jobs" instead of "Job Boards"), add that name to the list.
- **Skipped rows**: blank names, duplicate names (the first one wins), and anything listed in `exclude`.
- **Dates**: the `Event Date` text is shown as written. If it's empty, the `Date` field is formatted instead. The end of `Date` decides when an event counts as past.
- **Description**, or `AI summary` if Description is empty, shows under the resource name and is searchable.

The script prints a warning for every row it couldn't place and for every `curation.json` entry with no matching Notion row (usually a renamed row). It refuses to write if the library would shrink by more than half. That usually means a database isn't shared with the integration; pass `--force` if the drop is real.

### Tests

```bash
node --test scripts/sync-notion.test.mjs
```
