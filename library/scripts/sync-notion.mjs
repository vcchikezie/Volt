#!/usr/bin/env node
// Pull the Volt Library databases from Notion into data/resources.json.
//
//   NOTION_TOKEN=secret_... node scripts/sync-notion.mjs            write data/resources.json
//   NOTION_TOKEN=secret_... node scripts/sync-notion.mjs --dry-run  report changes only
//
// Notion is the source of truth for what it has. data/curation.json fills the gaps Notion
// leaves (section, kind, topic, link, event details), sets the order, and lists rows to skip.
// A Notion value always wins over a curated one, so fixing something in Notion is enough.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "data/resources.json");
const CURATION = path.join(ROOT, "data/curation.json");

export const RESOURCES_DS = "1b5ba277-333f-4b91-af77-1aa173ac16b5";
export const CATEGORIES_DS = "d8c8de5d-7cb3-4b50-8c42-fb089690192d";
const NOTION_VERSION = "2025-09-03";

// Notion API

async function queryAll(dataSourceId, token) {
  const rows = [];
  let cursor;
  do {
    const res = await fetch(`https://api.notion.com/v1/data_sources/${dataSourceId}/query`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Notion-Version": NOTION_VERSION, "Content-Type": "application/json" },
      body: JSON.stringify({ page_size: 100, ...(cursor && { start_cursor: cursor }) }),
    });
    if (!res.ok) {
      const body = await res.text();
      const hint = res.status === 404 ? " (is the database shared with your integration? In Notion: ••• → Connections)" : "";
      throw new Error(`Notion ${res.status} for data source ${dataSourceId}${hint}\n${body}`);
    }
    const page = await res.json();
    rows.push(...page.results);
    cursor = page.has_more ? page.next_cursor : null;
  } while (cursor);
  return rows;
}

// Property readers

const text = p => (p?.title ?? p?.rich_text ?? []).map(t => t.plain_text).join("").trim();
const prop = (page, name) => page.properties?.[name];

function readResource(page) {
  const date = prop(page, "Date")?.date;
  return {
    name: text(prop(page, "Name")).replace(/\s+/g, " "),
    url: (prop(page, "URL")?.url || "").trim(),
    tags: (prop(page, "Tags")?.multi_select || []).map(o => o.name),
    categoryIds: (prop(page, "Category")?.relation || []).map(r => r.id),
    eventText: text(prop(page, "Event Date")),
    dateStart: date?.start?.slice(0, 10) || "",
    dateEnd: (date?.end || date?.start || "").slice(0, 10),
    location: text(prop(page, "Location")),
    cost: text(prop(page, "Cost")),
    description: text(prop(page, "Description")) || text(prop(page, "AI summary")),
  };
}

// "Oct 19, 2024" or "Oct 23–24, 2024" or "Sep 30 – Oct 2, 2024"
export function formatWhen(start, end) {
  if (!start) return "";
  const d = s => new Date(s + "T12:00:00Z");
  const m = s => d(s).toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
  const day = s => d(s).getUTCDate();
  const yr = s => d(s).getUTCFullYear();
  if (!end || end === start) return `${m(start)} ${day(start)}, ${yr(start)}`;
  if (yr(start) !== yr(end)) return `${m(start)} ${day(start)}, ${yr(start)} – ${m(end)} ${day(end)}, ${yr(end)}`;
  if (m(start) !== m(end)) return `${m(start)} ${day(start)} – ${m(end)} ${day(end)}, ${yr(end)}`;
  return `${m(start)} ${day(start)}–${day(end)}, ${yr(end)}`;
}

const norm = s => s.toLowerCase().replace(/[‐-―]/g, "-").replace(/\s+/g, " ").trim();
const clean = o => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== "" && v != null && !(Array.isArray(v) && !v.length)));

// Merge Notion rows with the curation file. Pure, so it can be tested without Notion.
export function build(resourcePages, categoryPages, curation, today) {
  const warnings = [];

  const sectionByNotionName = new Map();
  for (const s of curation.sections) for (const n of s.notion || [s.name]) sectionByNotionName.set(norm(n), s.key);
  const categoryName = new Map(categoryPages.map(p => [p.id, text(prop(p, "Name"))]));

  const curated = new Map(Object.entries(curation.resources).map(([name, r], i) => [norm(name), { ...r, name, rank: i }]));
  const excluded = new Set((curation.exclude || []).map(norm));
  const seen = new Set();
  const out = [];

  for (const page of resourcePages) {
    if (page.in_trash || page.archived) continue;
    const n = readResource(page);
    if (!n.name) continue;
    const key = norm(n.name);
    if (excluded.has(key)) continue;
    if (seen.has(key)) { warnings.push(`Duplicate row skipped: "${n.name}"`); continue; }
    seen.add(key);

    const c = curated.get(key) || {};

    const notionSections = n.categoryIds.map(id => categoryName.get(id)).filter(Boolean);
    let section = null;
    for (const cat of notionSections) {
      section = sectionByNotionName.get(norm(cat));
      if (section) break;
      warnings.push(`"${n.name}": Notion category "${cat}" has no section. Add it to a section's "notion" list in data/curation.json.`);
    }
    section ||= c.section;
    if (!section) { warnings.push(`Not published, no section: "${n.name}". Set its Category in Notion or add it to data/curation.json.`); continue; }

    const topics = [...new Set(n.tags.map(t => curation.tagTopics[t]).filter(Boolean))];

    let event;
    if (section === "events") {
      const ce = c.event || {};
      event = clean({
        when: n.eventText || (n.dateStart && formatWhen(n.dateStart, n.dateEnd)) || ce.when,
        end: n.dateEnd || ce.end,
        location: n.location || ce.location,
        cost: n.cost || ce.cost,
      });
    }

    out.push({
      rank: c.rank ?? Infinity,
      ...clean({
        name: c.name || n.name,
        section,
        kind: c.kind,
        topics: topics.length ? topics : c.topics || ["g"],
        url: n.url || c.url,
        description: n.description || c.description,
      }),
      ...(event && { event }),
    });
  }

  for (const [key, c] of curated) if (!seen.has(key)) warnings.push(`In data/curation.json but not in Notion: "${c.name}"`);

  const order = new Map(curation.sections.map((s, i) => [s.key, i]));
  // Curated rows keep their order; new rows go after them, by section, then name.
  out.sort((a, b) => (a.rank - b.rank) || order.get(a.section) - order.get(b.section) || a.name.localeCompare(b.name));

  return {
    data: {
      synced: today,
      topics: curation.topics,
      sections: curation.sections.map(({ notion, ...s }) => s),
      resources: out.map(({ rank, ...r }) => r),
    },
    warnings,
  };
}

function diff(before, after) {
  const key = r => r.name;
  const a = new Map((before?.resources || []).map(r => [key(r), JSON.stringify(r)]));
  const b = new Map(after.resources.map(r => [key(r), JSON.stringify(r)]));
  return {
    added: [...b.keys()].filter(k => !a.has(k)),
    removed: [...a.keys()].filter(k => !b.has(k)),
    changed: [...b.keys()].filter(k => a.has(k) && a.get(k) !== b.get(k)),
  };
}

async function main() {
  const dryRun = process.argv.includes("--dry-run");
  const token = process.env.NOTION_TOKEN;
  if (!token) {
    console.error("Set NOTION_TOKEN to your Notion integration secret. See README.md.");
    process.exit(1);
  }

  const curation = JSON.parse(fs.readFileSync(CURATION, "utf8"));
  const [resources, categories] = await Promise.all([queryAll(RESOURCES_DS, token), queryAll(CATEGORIES_DS, token)]);
  const { data, warnings } = build(resources, categories, curation, new Date().toISOString().slice(0, 10));

  const before = fs.existsSync(OUT) ? JSON.parse(fs.readFileSync(OUT, "utf8")) : null;
  const { added, removed, changed } = diff(before, data);

  for (const w of warnings) console.warn(`! ${w}`);
  console.log(`\n${resources.length} Notion rows → ${data.resources.length} resources`);
  for (const [label, list] of [["Added", added], ["Removed", removed], ["Changed", changed]]) {
    if (list.length) console.log(`${label} (${list.length}): ${list.join(", ")}`);
  }

  // A sync that would empty most of the library is far more likely a sharing or API problem than real edits.
  if (before && data.resources.length < before.resources.length / 2 && !process.argv.includes("--force")) {
    console.error(`\nRefusing to write: resources would drop from ${before.resources.length} to ${data.resources.length}. Re-run with --force if that's intended.`);
    process.exit(1);
  }

  if (dryRun) { console.log("\nDry run, nothing written."); return; }
  fs.writeFileSync(OUT, JSON.stringify(data, null, 2) + "\n");
  console.log(`\nWrote ${path.relative(ROOT, OUT)}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(err => { console.error(err.message); process.exit(1); });
}
