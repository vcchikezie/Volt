// node --test scripts/
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { build, formatWhen } from "./sync-notion.mjs";

const curation = JSON.parse(fs.readFileSync(new URL("../data/curation.json", import.meta.url)));
const current = JSON.parse(fs.readFileSync(new URL("../data/resources.json", import.meta.url)));

const rt = s => (s ? [{ plain_text: s }] : []);
function page(name, p = {}) {
  return {
    id: p.id || name,
    properties: {
      Name: { title: rt(name) },
      URL: { url: p.url || null },
      Tags: { multi_select: (p.tags || []).map(n => ({ name: n })) },
      Category: { relation: (p.cats || []).map(id => ({ id })) },
      "Event Date": { rich_text: rt(p.eventText) },
      Date: { date: p.date || null },
      Location: { rich_text: rt(p.location) },
      Cost: { rich_text: rt(p.cost) },
      Description: { rich_text: rt(p.description) },
      "AI summary": { rich_text: [] },
    },
  };
}
const cat = (id, name) => ({ id, properties: { Name: { title: rt(name) } } });

test("bare Notion rows plus curation reproduce the current resources.json", () => {
  const pages = Object.keys(curation.resources).map(n => page(n));
  const { data, warnings } = build(pages, [], curation, current.synced);
  assert.deepEqual(warnings, []);
  assert.deepEqual(data, current);
});

test("Notion values win over curation, and gaps are reported", () => {
  const pages = [
    page("DEF CON 33", { cats: ["e"], tags: ["Cybersecurity"], date: { start: "2026-08-06", end: "2026-08-09" }, location: "Las Vegas", url: "https://defcon.org/" }),
    page("Slack", { url: "https://slack.com/intl/", description: "Team chat." }),
    page("Slack"),
    page("Certifications"),
    page(""),
    page("Mystery row"),
    page("Odd one", { cats: ["x"] }),
  ];
  const { data, warnings } = build(pages, [cat("e", "Events"), cat("x", "Misc")], curation, "2026-10-29");

  const defcon = data.resources.find(r => r.name === "DEF CON 33");
  assert.deepEqual(defcon, { name: "DEF CON 33", section: "events", topics: ["c"], url: "https://defcon.org/", event: { when: "Aug 6–9, 2026", end: "2026-08-09", location: "Las Vegas" } });

  const slack = data.resources.find(r => r.name === "Slack");
  assert.equal(slack.url, "https://slack.com/intl/");
  assert.equal(slack.description, "Team chat.");
  assert.equal(slack.kind, "Team chat");

  assert.deepEqual(data.resources.map(r => r.name), ["Slack", "DEF CON 33"]); // curated rows first
  assert.ok(warnings.includes('Duplicate row skipped: "Slack"'));
  assert.ok(warnings.some(w => w.startsWith('Not published, no section: "Mystery row"')));
  assert.ok(warnings.some(w => w.includes('Notion category "Misc" has no section')));
  assert.ok(warnings.includes('In data/curation.json but not in Notion: "Trello"'));
});

test("formatWhen", () => {
  assert.equal(formatWhen("2024-10-19"), "Oct 19, 2024");
  assert.equal(formatWhen("2024-10-23", "2024-10-24"), "Oct 23–24, 2024");
  assert.equal(formatWhen("2024-09-30", "2024-10-02"), "Sep 30 – Oct 2, 2024");
  assert.equal(formatWhen("2024-12-30", "2025-01-02"), "Dec 30, 2024 – Jan 2, 2025");
});
