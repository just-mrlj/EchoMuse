// Tests for what the Status tab's Wake word row reads: _wakeWordsRow(), and
// the fleet view's list of Echoes whose wake words Home Assistant chose.
//
//     node controller/tests/wake_words_row.test.mjs
//
// Home Assistant's two wake word pickers choose which wake words an Echo
// listens for, apart from the dashboard's model (#112). The dashboard has to
// show that, or the fleet view says one thing while the Echo does another.
//
// Absence of `wakeWords` is not a measurement: a dashboard served by a
// controller that has never heard of it keeps today's reading, the configured
// model, rather than claiming anything about who chose it.

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const HERE = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(HERE, "..", "static", "dashboard.jsx"), "utf8");
function lift(name) {
  const start = src.indexOf(`function ${name}`);
  if (start < 0) throw new Error(`dashboard.jsx no longer defines ${name}()`);
  let depth = 0, i = src.indexOf("{", start);
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") { depth--; if (depth === 0) break; }
  }
  return src.slice(start, i + 1);
}
const wakeWordsRow = eval(`(() => { ${lift("wwModelLabel")}; return ${lift("_wakeWordsRow")}; })()`);
const haDevices = eval(`(${lift("_haWakeWordDevices")})`);

let failed = 0;
const check = (name, cond) => { if (!cond) { failed++; console.error(`FAIL: ${name}`); } };
const jarvis = { id: "hey_jarvis_v0.1", name: "hey jarvis" };
const robot  = { id: "robot", name: "hey robot" };

const old = wakeWordsRow({}, { owwModel: "alexa_v0.1" });
check("no wakeWords reads the configured model", old.label === "alexa");
check("no wakeWords claims nothing", old.color === null && old.title === null);

const dash = wakeWordsRow({ wakeWords: { source: "dashboard", active: [jarvis], dropped: [], max: 4 } }, {});
check("the dashboard's choice is just the name", dash.label === "hey jarvis");
check("nothing dropped needs no explanation", dash.title === null);

const ha = wakeWordsRow({ wakeWords: { source: "ha", active: [jarvis, robot], dropped: [], max: 4 } }, {});
check("HA's choice names every wake word and who chose",
      ha.label === "hey jarvis + hey robot · chosen in Home Assistant");
check("HA's choice is not a warning by itself", ha.color === null);

const over = wakeWordsRow({ wakeWords: { source: "ha", active: [jarvis],
  dropped: [{ id: "robot", name: "hey robot", reason: "over_max" }], max: 1 } }, {});
check("a wake word not applied is counted", over.label.endsWith("· 1 not applied"));
check("and says why", over.title.includes("hey robot: not applied: this Echo listens for 1 at once"));
check("and is amber", over.color === "var(--warn)");

const gone = wakeWordsRow({ wakeWords: { source: "ha", active: [jarvis],
  dropped: [{ id: "x", name: "x", reason: "not_installed" }], max: 4 } }, {});
check("a vanished model says so", gone.title.includes("x: not applied: no longer installed"));

const off = wakeWordsRow({ wake_word: false, wakeWords: { source: "ha", active: [robot] } }, {});
check("No wake word wins over the list", off.label === "Off (Home Assistant)");

const fleet = haDevices([
  { label: "Kitchen", wakeWords: { source: "ha", active: [jarvis, robot] } },
  { label: "Office",  wakeWords: { source: "dashboard", active: [jarvis] } },
  { device_id: "G090X", wakeWords: { source: "ha", active: [robot] } },
  { label: "Old controller" },
]);
check("the fleet view lists only Echoes HA chose for",
      JSON.stringify(fleet) === JSON.stringify([
        { label: "Kitchen", names: "hey jarvis + hey robot" },
        { label: "G090X", names: "hey robot" }]));
check("no devices is no list", haDevices(undefined).length === 0);

if (failed) { console.error(`${failed} failed`); process.exit(1); }
console.log("wake_words_row: all ok");
