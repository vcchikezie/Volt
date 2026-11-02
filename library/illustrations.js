// Editorial thumbnail illustrations, drawn in code.
// Every scene is a view through an office window: one scene per section (what happens in the room),
// coloured and marked by the resource's topic (blind, walls, jacket and a wall object), with paper grain.
// Skin tone, hair and a mirrored layout vary per resource so a section does not repeat one picture.
// window.voltIllustration(item) returns an object URL for an SVG image, cached per variant.
(function () {
  const W = 400, H = 300;

  // Topic colours, softened for illustration: a = blind, d = jacket and marks, l = room walls
  const TOPIC = {
    c: { a: "#c4473c", d: "#8f3028", l: "#e9c2b8" },  // Cybersecurity: red
    p: { a: "#3f6fb5", d: "#2b4f86", l: "#b7cbe8" },  // Privacy: blue
    a: { a: "#66666f", d: "#45454c", l: "#c5c5cc" },  // AI: dark grey
    l: { a: "#2e2e2e", d: "#1c1c1c", l: "#a9a8a2" },  // Tech Law: black
    g: { a: "#bdbbb2", d: "#8e8c84", l: "#e2e0d8" },  // General: light grey
  };
  const SKIN = ["#e8b796", "#cf8f69", "#a8704c", "#6e4a33"];
  const HAIR = ["#2b211b", "#6e3f22", "#141210", "#9a5a34", "#4a4642"];
  const STONE = "#a39f94", STONE_D = "#8b877d", STONE_L = "#b8b4a9", FRAME = "#55534d", FRAME_D = "#3f3e3a";
  const SKY = "#9dcdee", SKY_L = "#bfe0f6", PAPER = "#f2efe7", PAPER_D = "#d8d3c7", WOOD = "#a86f3e", WOOD_D = "#7d5030";
  const ORANGE = "#d4663e";

  function seedOf(s) { let h = 2166136261; for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619); return h >>> 0; }

  // A seated (or standing) person whose hands rest at y = 0, centred on x = 0.
  function person(o) {
    const { skin, hair, jacket, style, headphones, hands = "desk" } = o;
    let hairBack = "", hairTop = "";
    if (style === 0) { hairTop = `<circle cx="0" cy="-95" r="12.5" fill="${hair}"/><circle cx="-1" cy="-109" r="7" fill="${hair}"/>`; }
    else if (style === 1) { hairTop = `<path d="M-13 -86 Q-14 -106 0 -107 Q14 -106 13 -86 Q8 -97 0 -97 Q-8 -97 -13 -86Z" fill="${hair}"/>`; }
    else { hairBack = `<rect x="-15" y="-98" width="30" height="34" rx="12" fill="${hair}"/>`; hairTop = `<path d="M-13 -84 Q-15 -106 0 -107 Q15 -106 13 -84 Q6 -98 -2 -96 Q-9 -94 -13 -84Z" fill="${hair}"/>`; }
    const arms = hands === "desk"
      ? `<path d="M-25 -46 L-14 -48 L4 -7 L-9 -2Z" fill="${jacket}"/><path d="M25 -46 L14 -48 L9 -7 L22 -3Z" fill="${jacket}"/>
         <path d="M-25 -46 L-19 -47 L-3 -5 L-9 -2Z" fill="#000" opacity=".12"/>
         <ellipse cx="-3" cy="-4" rx="7" ry="4.5" fill="${skin}"/><ellipse cx="14" cy="-4" rx="7" ry="4.5" fill="${skin}"/>`
      : `<path d="M-25 -46 L-14 -48 L-6 -14 L-18 -12Z" fill="${jacket}"/><path d="M25 -46 L14 -48 L6 -14 L18 -12Z" fill="${jacket}"/>
         <ellipse cx="-10" cy="-13" rx="6" ry="5" fill="${skin}"/><ellipse cx="10" cy="-13" rx="6" ry="5" fill="${skin}"/>`;
    const phones = headphones
      ? `<path d="M-14 -86 Q-15 -112 0 -112 Q15 -112 14 -86" fill="none" stroke="${FRAME_D}" stroke-width="3.5"/><rect x="-18" y="-90" width="7" height="12" rx="3" fill="${FRAME_D}"/><rect x="11" y="-90" width="7" height="12" rx="3" fill="${FRAME_D}"/>`
      : "";
    return `${hairBack}
      <path d="M-31 0 L-27 -45 Q-23 -63 0 -65 Q23 -63 27 -45 L31 0Z" fill="${jacket}"/>
      <path d="M2 -64 Q23 -62 27 -45 L31 0 L12 0Z" fill="#000" opacity=".14"/>
      <path d="M-9 -63 L0 -42 L9 -63Z" fill="${PAPER}"/>
      <rect x="-5" y="-72" width="10" height="10" fill="${skin}"/><rect x="-5" y="-66" width="10" height="4" fill="#000" opacity=".15"/>
      <ellipse cx="0" cy="-84" rx="12" ry="14" fill="${skin}"/>
      <path d="M1 -98 Q12 -96 12 -84 Q12 -72 1 -70Z" fill="#000" opacity=".1"/>
      ${hairTop}${phones}${arms}`;
  }

  const chair = (x, y) => `<rect x="${x - 44}" y="${y - 80}" width="46" height="78" rx="9" fill="#4b4a46"/><rect x="${x - 44}" y="${y - 80}" width="10" height="78" rx="5" fill="#000" opacity=".15"/>`;
  const desk = () => `<rect x="70" y="188" width="222" height="12" fill="${WOOD}"/><rect x="70" y="198" width="222" height="40" fill="${WOOD_D}"/><rect x="70" y="198" width="222" height="4" fill="#000" opacity=".15"/>`;

  // What is on the back wall, on the desk and in front of the window, per section
  const SECTION = {
    laws: {
      desk: () => `<path d="M88 185 L132 180 L138 188 L94 193Z" fill="${ORANGE}"/><path d="M94 193 L138 188 L138 191 L95 196Z" fill="#a8482a"/>`,
      front: () => `<path d="M168 190 C200 192 212 202 222 238 C234 276 250 286 236 300 L262 300 C276 282 260 262 250 238 C240 204 224 192 196 187Z" fill="${PAPER}"/>
        <path d="M222 238 C234 276 250 286 236 300 L244 300 C256 284 242 268 232 238Z" fill="${PAPER_D}"/>
        <path d="M-10 286 C40 250 70 300 118 276 C150 260 170 282 200 300 L170 300 C150 290 140 278 120 290 C80 312 40 268 -10 302Z" fill="${PAPER}"/>
        <path d="M338 120 C360 100 390 118 410 104 L410 120 C390 134 362 116 346 134Z" fill="${PAPER}"/>`,
    },
    regulators: {
      wall: (t) => `<circle cx="112" cy="128" r="22" fill="${t.d}"/><circle cx="112" cy="128" r="16" fill="none" stroke="${PAPER}" stroke-width="1.5"/>
        <path d="M100 124 L112 116 L124 124Z" fill="${PAPER}"/><rect x="102" y="126" width="3" height="10" fill="${PAPER}"/><rect x="110.5" y="126" width="3" height="10" fill="${PAPER}"/><rect x="119" y="126" width="3" height="10" fill="${PAPER}"/><rect x="100" y="137" width="24" height="3" fill="${PAPER}"/>`,
      desk: (t) => `<rect x="92" y="176" width="40" height="12" fill="${PAPER}"/><rect x="92" y="172" width="40" height="5" fill="${PAPER_D}"/>
        <rect x="236" y="176" width="22" height="12" rx="2" fill="${t.d}"/><rect x="243" y="160" width="8" height="17" rx="3" fill="#4b4a46"/><circle cx="247" cy="158" r="6" fill="#4b4a46"/>`,
    },
    frameworks: {
      wall: (t) => `<rect x="76" y="102" width="78" height="66" fill="${PAPER}"/><path d="M76 116 H154 M76 130 H154 M76 144 H154 M76 158 H154 M90 102 V168 M104 102 V168 M118 102 V168 M132 102 V168 M146 102 V168" stroke="${PAPER_D}" stroke-width="1"/>
        <path d="M95 117 L115 135 L137 117 M115 135 L115 155" stroke="${FRAME}" stroke-width="2" fill="none"/>
        <rect x="88" y="110" width="14" height="12" fill="${t.a}"/><rect x="130" y="110" width="14" height="12" fill="${t.a}"/><rect x="108" y="129" width="14" height="12" fill="${FRAME}"/><rect x="108" y="150" width="14" height="12" fill="${t.d}"/>`,
      desk: () => `<path d="M88 185 L132 180 L138 188 L94 193Z" fill="${PAPER}"/><path d="M236 176 h44 v12 h-44z" fill="${PAPER_D}"/>`,
    },
    certifications: {
      wall: (t) => `<rect x="80" y="104" width="66" height="50" fill="#6b5a3e"/><rect x="85" y="109" width="56" height="40" fill="${PAPER}"/>
        <path d="M93 119 H133 M97 126 H129 M97 132 H125" stroke="${PAPER_D}" stroke-width="2"/>
        <path d="M126 140 L122 158 L127 154 L131 160 L132 142Z" fill="${t.d}"/><circle cx="127" cy="139" r="7" fill="${t.a}"/><circle cx="127" cy="139" r="3.5" fill="${PAPER}"/>`,
      desk: (t) => `<rect x="150" y="150" width="34" height="26" fill="${PAPER}" transform="rotate(-6 167 163)"/><circle cx="176" cy="170" r="4" fill="${t.a}"/>`,
    },
    learning: {
      wall: () => `<rect x="76" y="102" width="80" height="54" fill="#34403b"/><rect x="76" y="156" width="80" height="4" fill="#6b5a3e"/>
        <path d="M86 116 H116 M86 126 H130 M86 136 H110" stroke="#dfe3dc" stroke-width="2" opacity=".85"/><circle cx="140" cy="122" r="8" fill="none" stroke="#dfe3dc" stroke-width="2" opacity=".85"/>`,
      desk: (t) => `<rect x="88" y="179" width="44" height="9" fill="${t.a}"/><rect x="91" y="171" width="40" height="8" fill="${ORANGE}"/><rect x="86" y="164" width="42" height="7" fill="${PAPER_D}"/>
        <path d="M214 186 L270 186 L276 190 L208 190Z" fill="#6c6b66"/><path d="M220 186 L224 150 L274 150 L270 186Z" fill="#8f8e88"/><circle cx="247" cy="168" r="3" fill="${PAPER}"/>`,
    },
    events: {
      standing: true,
      wall: () => `<path d="M150 92 L200 92 L250 238 L100 238Z" fill="#fff" opacity=".18"/>`,
      desk: (t) => `<path d="M138 170 L206 170 L198 238 L146 238Z" fill="${t.a}"/><path d="M138 170 L206 170 L205 178 L139 178Z" fill="${t.d}"/>
        <path d="M188 170 L198 150" stroke="${FRAME_D}" stroke-width="2.5"/><ellipse cx="199" cy="148" rx="4" ry="6" fill="${FRAME_D}"/>`,
      front: () => {
        let s = "";
        for (let i = 0; i < 9; i++) { const x = 18 + i * 46 + (i % 2) * 8, y = 270 + (i % 3) * 6; s += `<ellipse cx="${x}" cy="${y + 32}" rx="26" ry="22" fill="#3d3c39"/><circle cx="${x}" cy="${y}" r="13" fill="#2f2e2b"/>`; }
        return s;
      },
    },
    research: {
      wall: (t) => `<rect x="76" y="102" width="78" height="62" fill="${PAPER}"/><path d="M84 156 H148" stroke="${FRAME}" stroke-width="1.5"/>
        <rect x="88" y="136" width="10" height="20" fill="${t.a}"/><rect x="103" y="122" width="10" height="34" fill="${FRAME}"/><rect x="118" y="128" width="10" height="28" fill="${t.d}"/><rect x="133" y="112" width="10" height="44" fill="${ORANGE}"/>`,
      desk: () => `<path d="M90 186 L134 181 L138 188 L94 192Z" fill="${PAPER}"/><circle cx="214" cy="172" r="12" fill="#cfe6f3" stroke="${FRAME_D}" stroke-width="3.5"/><path d="M222 181 L236 194" stroke="${FRAME_D}" stroke-width="5" stroke-linecap="round"/>`,
    },
    news: {
      hands: "up",
      desk: () => `<rect x="134" y="136" width="62" height="46" fill="${PAPER}"/><rect x="164" y="136" width="2" height="46" fill="${PAPER_D}"/>
        <rect x="139" y="142" width="22" height="7" fill="${FRAME}"/><path d="M139 155 H160 M139 161 H160 M139 167 H160 M139 173 H156 M170 142 H191 M170 148 H191 M170 154 H191 M170 160 H186" stroke="${PAPER_D}" stroke-width="2"/>`,
      front: () => `<rect x="346" y="96" width="30" height="22" fill="${PAPER}" transform="rotate(-18 361 107)"/><rect x="360" y="160" width="26" height="20" fill="${PAPER}" transform="rotate(14 373 170)"/><rect x="330" y="222" width="28" height="20" fill="${PAPER}" transform="rotate(-8 344 232)"/>`,
    },
    podcasts: {
      headphones: true,
      wall: (t) => `<path d="M92 120 Q84 132 92 144 M84 114 Q72 132 84 150 M76 108 Q60 132 76 156" stroke="${t.d}" stroke-width="3" fill="none" stroke-linecap="round"/>`,
      desk: () => `<path d="M300 170 L250 166 L216 140" stroke="${FRAME_D}" stroke-width="4" fill="none" stroke-linejoin="round"/><rect x="200" y="132" width="16" height="28" rx="8" fill="#3a3936"/><path d="M203 140 H213 M203 146 H213 M203 152 H213" stroke="#77756f" stroke-width="1.5"/>`,
    },
    books: {
      hands: "up",
      wall: (t) => {
        const cols = [t.a, FRAME, ORANGE, PAPER_D, t.d, "#6b5a3e", t.a, PAPER, FRAME_D];
        let s = `<rect x="72" y="100" width="84" height="80" fill="#6b5a3e"/>`;
        [104, 130, 156].forEach((y, row) => {
          s += `<rect x="76" y="${y}" width="76" height="22" fill="#4f4230"/>`;
          let x = 78;
          for (let i = 0; x < 146; i++) { const w = 6 + ((i * 7 + row * 3) % 5), h = 15 + ((i * 5 + row) % 6); s += `<rect x="${x}" y="${y + 22 - h}" width="${w}" height="${h}" fill="${cols[(i + row * 2) % cols.length]}"/>`; x += w + 1; }
          s += `<rect x="74" y="${y + 22}" width="80" height="4" fill="#3a3024"/>`;
        });
        return s;
      },
      desk: () => `<path d="M138 168 L166 160 L166 182 L138 188Z" fill="${PAPER}"/><path d="M194 168 L166 160 L166 182 L194 188Z" fill="${PAPER_D}"/>`,
    },
    tools: {
      wall: () => {
        let s = `<rect x="76" y="102" width="80" height="62" fill="#c9a978"/>`;
        for (let y = 108; y < 162; y += 9) for (let x = 82; x < 154; x += 9) s += `<circle cx="${x}" cy="${y}" r="1.3" fill="#8f7550"/>`;
        return s + `<path d="M92 112 L92 150" stroke="${FRAME_D}" stroke-width="5" stroke-linecap="round"/><circle cx="92" cy="110" r="6" fill="none" stroke="${FRAME_D}" stroke-width="4"/>
          <circle cx="128" cy="132" r="12" fill="${FRAME}"/><circle cx="128" cy="132" r="4" fill="#c9a978"/>` +
          [0, 45, 90, 135, 180, 225, 270, 315].map(a => `<rect x="125" y="116" width="6" height="6" fill="${FRAME}" transform="rotate(${a} 128 132)"/>`).join("");
      },
      desk: (t) => `<rect x="222" y="168" width="48" height="20" rx="2" fill="${t.a}"/><rect x="222" y="168" width="48" height="5" fill="${t.d}"/><path d="M238 168 V160 H254 V168" fill="none" stroke="${FRAME_D}" stroke-width="3"/>`,
    },
    communities: {
      crowd: true,
      wall: () => `<rect x="80" y="104" width="50" height="26" rx="10" fill="${PAPER}"/><path d="M96 130 L92 140 L106 130Z" fill="${PAPER}"/><rect x="202" y="100" width="44" height="22" rx="9" fill="${PAPER}"/><path d="M230 122 L236 131 L222 122Z" fill="${PAPER}"/>`,
    },
    jobs: {
      wall: (t) => `<rect x="76" y="102" width="78" height="60" fill="#c49a6c"/>
        <rect x="84" y="110" width="22" height="18" fill="${PAPER}"/><rect x="112" y="108" width="20" height="22" fill="${t.l}"/><rect x="88" y="134" width="20" height="20" fill="${t.a}"/><rect x="118" y="136" width="26" height="18" fill="${PAPER}"/>
        <circle cx="95" cy="110" r="2.2" fill="${ORANGE}"/><circle cx="122" cy="108" r="2.2" fill="${FRAME_D}"/><circle cx="98" cy="134" r="2.2" fill="${ORANGE}"/><circle cx="131" cy="136" r="2.2" fill="${FRAME_D}"/>`,
      desk: () => `<rect x="218" y="164" width="52" height="24" rx="3" fill="#6b4126"/><rect x="218" y="174" width="52" height="3" fill="#4a2c19"/><path d="M234 164 V156 H254 V164" fill="none" stroke="#4a2c19" stroke-width="3"/>`,
    },
  };

  // A wall object that marks the topic, on the right of the room
  const MOTIF = {
    c: (t) => `<path d="M252 128 V118 Q252 104 266 104 Q280 104 280 118 V128" fill="none" stroke="${FRAME_D}" stroke-width="5"/><rect x="246" y="126" width="40" height="32" rx="4" fill="${t.d}"/><circle cx="266" cy="139" r="4" fill="${PAPER}"/><rect x="264.5" y="140" width="3" height="9" fill="${PAPER}"/>`,
    p: (t) => `<rect x="240" y="100" width="52" height="62" fill="${FRAME}"/><rect x="244" y="104" width="44" height="54" fill="${SKY}"/><rect x="244" y="104" width="44" height="46" fill="${t.a}"/>
      <path d="M244 112 H288 M244 120 H288 M244 128 H288 M244 136 H288 M244 144 H288" stroke="${t.d}" stroke-width="1.5"/><rect x="242" y="149" width="48" height="4" fill="${t.d}"/>`,
    a: (t) => `<path d="M246 108 L286 152 M286 108 L246 152 M266 100 V160" stroke="${t.d}" stroke-width="1.5" opacity=".6"/>
      <rect x="252" y="116" width="28" height="28" rx="3" fill="${t.d}"/><rect x="259" y="123" width="14" height="14" fill="${t.l}"/>` +
      [120, 127, 134, 141].map(y => `<rect x="247" y="${y}" width="5" height="2" fill="${t.d}"/><rect x="280" y="${y}" width="5" height="2" fill="${t.d}"/>`).join("") +
      `<circle cx="246" cy="108" r="3.5" fill="${t.d}"/><circle cx="286" cy="108" r="3.5" fill="${t.d}"/><circle cx="246" cy="152" r="3.5" fill="${t.d}"/><circle cx="286" cy="152" r="3.5" fill="${t.d}"/>`,
    l: (t) => `<rect x="264" y="108" width="4" height="48" fill="${t.d}"/><rect x="254" y="154" width="24" height="5" fill="${t.d}"/><rect x="242" y="110" width="48" height="3" fill="${t.d}"/>
      <path d="M246 113 L238 132 H254Z M286 113 L278 132 H294Z" fill="none" stroke="${t.d}" stroke-width="1.2"/><path d="M237 132 Q246 142 255 132Z M277 132 Q286 142 295 132Z" fill="${t.d}"/><circle cx="266" cy="106" r="4" fill="${t.d}"/>`,
    g: (t) => `<circle cx="266" cy="128" r="22" fill="${PAPER}" stroke="${FRAME}" stroke-width="4"/><path d="M266 128 V114 M266 128 L276 134" stroke="${FRAME_D}" stroke-width="2.5" stroke-linecap="round"/>` +
      [0, 90, 180, 270].map(a => `<rect x="265" y="108" width="2" height="5" fill="${FRAME}" transform="rotate(${a} 266 128)"/>`).join(""),
  };

  function grainDefs() {
    return `<filter id="grain" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" seed="7" stitchTiles="stitch"/>
      <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -2.2 1.35"/></filter>
      <filter id="speck" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="1.4" numOctaves="1" seed="3" stitchTiles="stitch"/>
      <feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 -2.6 1.45"/></filter>`;
  }

  function scene(section, topic, seed) {
    const t = TOPIC[topic] || TOPIC.g;
    const S = SECTION[section] || SECTION.laws;
    const skin = SKIN[seed % SKIN.length], hair = HAIR[(seed >>> 3) % HAIR.length], style = (seed >>> 6) % 3;
    const mirror = (seed >>> 9) % 2 === 1;
    const blindDrop = topic === "p" ? 38 : 0;  // privacy: the blind comes down further
    const fig = (x, deskY, scale, s2) => `<g transform="translate(${x} ${deskY}) scale(${scale})">${person({
      skin: SKIN[s2 % SKIN.length], hair: HAIR[(s2 >>> 2) % HAIR.length], style: (s2 >>> 4) % 3,
      jacket: t.d, headphones: S.headphones, hands: S.hands || (S.standing ? "low" : "desk") })}</g>`;

    let people;
    if (S.crowd) people = chair(108, 188) + chair(250, 188) + fig(108, 188, 0.86, seed + 17) + fig(250, 188, 0.86, seed + 41) + chair(178, 188) + fig(178, 188, 1, seed);
    else if (S.standing) people = fig(172, 176, 1, seed);
    else people = chair(172, 188) + fig(172, 188, 1, seed);

    const body = `
      <rect width="${W}" height="${H}" fill="${SKY}"/>
      <rect x="346" y="40" width="40" height="260" fill="${SKY_L}"/><rect x="372" y="80" width="40" height="220" fill="${SKY_L}" opacity=".7"/>
      <rect x="0" y="0" width="338" height="${H}" fill="${STONE}"/><rect x="322" y="0" width="16" height="${H}" fill="${STONE_D}"/>
      <rect x="50" y="10" width="260" height="236" fill="${FRAME}"/>
      <rect x="60" y="20" width="240" height="218" fill="${t.l}"/>
      <rect x="60" y="92" width="34" height="146" fill="#000" opacity=".06"/>
      ${S.wall ? S.wall(t) : ""}
      ${MOTIF[topic] ? MOTIF[topic](t) : ""}
      ${people}
      ${S.standing ? "" : desk()}
      ${S.desk ? S.desk(t) : ""}
      <rect x="60" y="20" width="240" height="${38 + blindDrop}" fill="${t.a}"/>
      <rect x="60" y="${50 + blindDrop}" width="240" height="8" fill="${t.d}"/>
      <rect x="60" y="20" width="240" height="${38 + blindDrop}" fill="#fff" opacity=".07"/>
      <rect x="50" y="64" width="260" height="11" fill="${FRAME}"/><rect x="50" y="74" width="260" height="3" fill="${FRAME_D}"/>
      <rect x="296" y="20" width="4" height="218" fill="${FRAME_D}"/>
      <rect x="38" y="236" width="284" height="16" fill="${STONE_L}"/><rect x="38" y="250" width="284" height="6" fill="${STONE_D}"/>
      ${S.front ? S.front(t) : ""}`;

    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="28 30 344 258" width="${W}" height="${H}">
      <defs>${grainDefs()}</defs>
      <g ${mirror ? `transform="translate(${W} 0) scale(-1 1)"` : ""}>${body}</g>
      <rect width="${W}" height="${H}" filter="url(#grain)" opacity=".12"/>
      <rect width="${W}" height="${H}" filter="url(#speck)" opacity=".1"/>
    </svg>`;
  }

  const cache = new Map();
  window.voltIllustration = function (it) {
    const seed = seedOf(it.n || it.call || "");
    const key = `${it.c}|${it.t[0]}|${seed % 120}`;
    if (!cache.has(key)) {
      const svg = scene(it.c, it.t[0], seed % 120 + (seed >>> 16));
      cache.set(key, URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" })));
    }
    return cache.get(key);
  };
  window.voltIllustrationSVG = (section, topic, seed) => scene(section, topic, seed);
})();
