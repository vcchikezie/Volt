# Typography

Counsel + Ledger: the authority of an editorial serif in the statement, the precision of a light grotesk and a monospace everywhere else. Three families, each with one job. All three are open-source Google Fonts. The families, weights and scale are unchanged from the October 2026 agreement.

| Family | Job | Weights | Tracking |
|---|---|---|---|
| **Newsreader** | Statements: hero lines, headlines, move titles. An editorial serif by Production Type with optical sizes, so it sharpens as it grows. | Light 300, italic for emphasis | −0.02em to −0.015em |
| **Inter** | Sublines and section heads in Light 300; everything a person reads in Regular 400; titles and captions in Medium 500. | 300, 400, 500, 600 | −0.02em on Light sublines; 0 on body; +0.02em on captions |
| **JetBrains Mono** | Labels, calls to action and identifiers. Capitals for labels and buttons, tracked +0.1em. | 400, 500, 700 | +0.1em on labels and buttons; +0.05em on identifiers |

## Scale (desktop and publication)

| Style | Size / line | Weight | Font | Tracking | Use |
|---|---|---|---|---|---|
| `display-01` | 88 / 90px | 300 | Newsreader | −0.02em | Hero statements, one per page or post |
| `display-02` | 64 / 68px | 300 | Newsreader | −0.02em | Secondary statements, slide and cover headlines |
| `h1` | 48 / 54px | 300 | Newsreader | −0.015em | Section headlines and page titles |
| `statement-s` | 32 / 38px | 300 | Newsreader | −0.015em | Move titles, card statements, closing lines |
| `h2` | 28 / 38px | 300 | Inter | −0.02em | Sublines under statements, section heads |
| `h3` | 20 / 28px | 500 | Inter | −0.01em | Card and list titles |
| `body-large` | 20 / 32px | 300 | Inter | 0 | Lead paragraphs |
| `body` | 16 / 26px | 400 | Inter | 0 | Default text, form labels, data rows, clauses |
| `caption` | 12 / 17px | 500 | Inter | +0.02em | Captions, helper text |
| `label` | 13 / 20px | 500 | JetBrains Mono | +0.1em | Eyebrows, section labels, audience tags, in capitals |
| `cta` | 14 / 20px | 700 | JetBrains Mono | +0.1em | Pill buttons, in capitals |
| `code` | 13 / 20px | 400 | JetBrains Mono | +0.05em | Identifiers, file numbers, hashes, URLs |

On phones, statements scale down to about 60% (`display-01` to 48px) and never below 32px.

## Emphasis

- One phrase per statement may be emphasised, usually the last: "Every hour on strategy, *not search.*"
- Emphasis is Newsreader **italic**, in the same `ink` as the rest of the statement: white on black grounds, black on white. The change of letterform is the emphasis.
- Never emphasise with bold, underline, colour or a gradient.

## Rules

- **Newsreader sets statements only,** at 32px and up. It never sets body copy, buttons or labels.
- **Inter carries every sentence a person reads,** legal clauses included. Light 300 only at 20px and up.
- **Mono marks labels, actions and the machine-true:** `MOVE 01 //`, `BOOK A SCOPING CALL`, file numbers, hashes, URLs. Labels and buttons are capitals, written in sentence case in the source and set in capitals by style, so screen readers read words, not letters.
- **Casing.** Statements and headings in sentence case. Product and lexicon nouns keep their capitals.
- **Numerals** in tables and metrics use Inter's tabular figures (`font-variant-numeric: tabular-nums`); hero figures stay proportional.
- **Black and white type.** Text is `ink` or `ink-muted`, always. No text is ever set in a signal colour: a status word stays `ink` beside its coloured dot.
- **The wordmark is not type.** Use the logo files.
