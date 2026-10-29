# Volt Library

This folder is the Volt Library website. All commands below run from inside `library/`.

A searchable directory of resources on cybersecurity, data privacy, AI and tech law: laws, regulators, frameworks, certifications, courses, events, research, news, podcasts, books, tools, communities and jobs.

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
| `data/catalogue.xlsx` | The catalogue. Source of truth for everything the site shows | Yes |
| `data/resources.json` | What the site reads. Generated from the spreadsheet | No, it gets overwritten |
| `scripts/import_catalogue.py` | Turns the spreadsheet into `resources.json` | |
| `index.html` | The site | Yes |

## Updating the content

1. Edit `data/catalogue.xlsx`. Its **Read Me** tab explains every column. The site uses the **Resources** and **Sections** tabs.
2. Run the import. It needs Python 3 and openpyxl (`pip install openpyxl`):

   ```bash
   python3 scripts/import_catalogue.py --dry-run
   ```

   This checks the spreadsheet and reports what would change, without writing anything. If it looks right, run it again without `--dry-run`.
3. Commit both `data/catalogue.xlsx` and `data/resources.json`.

The import stops without writing if a row is missing a required field (ID, Name, Section Key, Type, Topics, Description, URL), uses a section key that isn't on the Sections tab, uses an unknown topic, or has an invalid date. It warns about descriptions over 160 characters and events that have already ended. It also refuses to write if the library would shrink by more than half, which protects against importing the wrong file; pass `--force` if the drop is real.

### What the site shows from each row

- **Name** links to the **URL**.
- Under the name: **Type**, then **Jurisdiction** (unless Global), the **Status** of laws, **Publisher**, **Level** (unless All) and **Cost**. Events show their dates and **Location** instead of jurisdiction.
- **Description** appears below that, and **Topics** appear on the right.
- Search covers name, type, description, publisher, jurisdiction, level, section, location and dates.
- Events are hidden automatically once their **End Date** has passed. An event with no dates shows "Dates to be announced".
- A **Status** of Inactive shows "No longer updated".
- **Curator Notes**, **Origin**, **Format** and **Last Checked** are not shown. The newest Last Checked date appears in the header as "checked".
- "United States" and "European Union" in Jurisdiction are shown as US and EU.
