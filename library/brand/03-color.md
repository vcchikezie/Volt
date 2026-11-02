# Color

Go is played in black and white. The palette is the surface of a championship match: black slate stones (*kuro-ishi*), white clamshell stones (*shiro-ishi*) and the greys of the Kaya board's shadows. Black and white are the brand, on every surface, in the logo and in the Sente call to action. Red, yellow and traffic green are signals: subliminal, strategic indicators of a state and, on black granite panels, the colour of a call to action by purpose.

## Primary palette

| Token | Name | Hex | RGB | CMYK | Role |
|---|---|---|---|---|---|
| `kuro` | Kuro (Black) | #000000 | 0, 0, 0 | 0, 0, 0, 100 | The dark field, text on white, the logo on light, the Sente button on light |
| `shiro` | Shiro (White) | #FFFFFF | 255, 255, 255 | 0, 0, 0, 0 | The light field, cards, text on black, the logo on dark, the Sente button on dark |

## Neutrals (the board)

| Token | Name | Hex | RGB | Role |
|---|---|---|---|---|
| `graphite` | Graphite | #141414 | 20, 20, 20 | Cards and sheets on the black field |
| `charcoal` | Charcoal | #2A2A2A | 42, 42, 42 | Hairlines and Goban lines on black |
| `slate-gray` | Slate Gray | #6E6E73 | 110, 110, 115 | Secondary text on white |
| `stone-gray` | Stone Gray | #9A9A9E | 154, 154, 158 | Large meta, chart de-emphasis |
| `silver` | Silver | #B4B4B9 | 180, 180, 185 | Secondary text on black |
| `kaya-silver` | Kaya Silver | #E5E5EA | 229, 229, 234 | Hairlines and tracks on white |
| `paper` | Paper | #F7F7F7 | 247, 247, 247 | Image, sample and code tiles on white |

## The three signals

| Token | Name | Hex | RGB | CMYK | Means |
|---|---|---|---|---|---|
| `signal-green` | Signal Green (traffic green) | #00A651 | 0, 166, 81 | 85, 10, 100, 0 | **Go.** Complete, verified, live, on track |
| `signal-yellow` | Signal Yellow (traffic amber) | #FFC20E | 255, 194, 14 | 0, 25, 100, 0 | **Caution.** At risk, pending review, needs attention |
| `signal-red` | Signal Red (traffic red) | #E0261F | 224, 38, 31 | 5, 95, 100, 0 | **Stop.** Breach, failed, non-compliant |

CMYK builds are starting points for coated stock; proof them.

**The signal rules.**

- **Two uses only.** A signal colour marks a state: a dot in a status flag, a progress stone, a figure that has crossed a threshold, the last-move marker on the decisive stone of a drawn Goban. Or it fills a signal button on a granite panel: green to commit, yellow to read, red when time or risk presses (see **Signal granite**). It is never text, a background, a heading, an emphasis, a glow or a decoration.
- **Subliminal.** State marks are small: a dot, a ring, a stone. A reader should feel the state before they read it, and never notice the colour as styling. Signal buttons are the one larger use, and they live only on black granite panels, one or two per page.
- **Strategic.** Use a signal only where the state changes what the reader does next. If nothing changes, leave it black and white.
- **Never alone.** Every state mark sits beside a word ("Verified", "At risk", "Lapsed"), and every signal button carries a label that names the move, so the meaning survives colour-blindness and greyscale print. Green and red are only 1.5:1 apart in lightness: the word carries the state, never the colour.
- **In order.** The three appear together, in traffic order (green, yellow, red), only in a status key or legend, or as a row of signal buttons on a granite panel. Elsewhere in the interface each state mark appears alone.
- No blue, no purple, no gradients between colours.

**Signal granite** (approved October 2026). Black panels with a granite matrix (1px white flecks on a 4px pitch, 0–10% opacity) carry the signal buttons: `go`, `caution` and `stop`. The full rules are in **Signal granite**.

## Roles and themes

Components consume roles, not palette colours. Light is the default for content and documents; marketing leads with Dark.

| Role | Light | Dark | Use |
|---|---|---|---|
| `field` | White | Black | Page ground (60%) |
| `card` | White | Graphite | Cards, sheets, the app header (30%), always with a hairline |
| `surface` | Paper | Graphite | Image, sample and code tiles |
| `hairline` | Kaya Silver | Charcoal | 1px borders and dividers (30%) |
| `ink` | Black | White | All primary text, selection, the logo |
| `ink-muted` | Slate Gray | Silver | Sublines, captions, meta, secondary labels |
| `sente` / `on-sente` | Black / White | White / Black | The primary action: the one filled pill per view |
| `status-go` / `status-caution` / `status-stop` | Green / Yellow / Red | Green / Yellow / Red | State marks, each beside a word |
| `goban-line` | Black at 3% | White at 3% | Grid texture |

The Sente button is black with white text on the white field and inverts to white with black text on the black field, so the primary action is always the highest-contrast element on the screen, and never coloured. Signal buttons are their own variants for granite panels, not a coloured Sente.

## The 60-30-10 balance

- **60% field:** white in Light, black in Dark.
- **30% structure:** cards, hairlines, tabular layouts, the neutrals.
- **10% ink accents:** the Sente pill, the logo, selected states, the emphasised phrase. Signals sit inside this 10%: state marks are a small fraction of it, and the signal buttons on a granite panel count here too.

A screen with no signal colour at all is on-brand. A screen where the signal marks are what you notice first is not; on a granite panel, the one `go` button may be.

## Contrast

Measured WCAG 2 ratios for the pairs the system uses:

| Pair | Ratio | Verdict |
|---|---|---|
| `ink` on `field` (both themes) | 21.0 | AAA |
| White on Graphite (dark `card`) | 18.4 | AAA |
| `on-sente` on `sente` | 21.0 | AAA |
| Slate Gray on white / paper (`ink-muted`, Light) | 5.1 / 4.7 | AA |
| Silver on black / graphite (`ink-muted`, Dark) | 10.2 / 8.9 | AAA |
| Stone Gray on white | 2.8 | Marks, or text at 24px+ / bold 19px+ |
| Stone Gray on black | 7.5 | AAA |
| Signal Green on white / on black | 3.2 / 6.6 | Marks: passes 3:1 for graphics on both |
| Signal Yellow on white / on black | 1.6 / 13.0 | On white, a yellow mark takes a 1px `ink` ring; on black it stands alone |
| Signal Red on white / on black | 4.7 / 4.5 | Marks on both |
| Signal Green vs Signal Red | 1.5 | **Always carry the word** |
| Black on Signal Green (`go` label) | 6.6 | AA |
| Black on Signal Yellow (`caution` label) | 13.0 | AAA |
| White on Signal Red (`stop` label) | 4.7 | AA |
| White / Silver on the brightest granite fleck | 17.4 / 8.4 | AAA |

Signal colours are never used for text. On a signal button the label is black or white, at the ratios above.

## Glow, rules and emphasis (Counsel + Ledger)

- **Lead with dark.** Marketing surfaces (social posts, covers, the website hero, slides) default to the Dark theme: black ground, white text. Content pages and documents stay Light.
- **The glow.** On black grounds, one soft radial glow may sit behind the hero art or the mono Goban: `glow-shiro` (white at 8%) fading to transparent by about 75%. It reads as light falling on the board, not as a colour. Never on white grounds, never behind body text, never on the logo.
- **Display rules.** On black grounds, the rule between a statement and its CTA row is `rule-on-dark` (white at 24%). Structural borders stay `hairline`.
- **Emphasis.** The italic phrase in a statement is `ink`, like the rest of it: the italic does the work.
- **Calls to action.** The primary pill is `sente`: black on white grounds, white on black grounds, with its label in JetBrains Mono capitals. A second action is an outline pill or an arrow link. On a granite panel, the actions are signal buttons by purpose.
- **Granite.** One or two black granite panels per marketing page or post, each with its own texture only: never with the glow or the mono Goban on the same panel.
