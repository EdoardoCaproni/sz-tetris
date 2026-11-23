// Replays the cases written by test_engine_js.py with viewer/engine.js and
// prints the number of mismatches (0 = same rules, keys and features).
'use strict';
const fs = require('fs');
const path = require('path');
const E = require(path.join(__dirname, '..', 'viewer', 'engine.js'));

const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let bad = 0;
for (const c of cases) {
  const pl = E.placements(c.W);
  const full = (1 << c.W) - 1;
  if (c.key !== undefined && E.key(c.rows, c.W) !== c.key) { bad++; console.log('key', c); }
  const f = E.features(c.rows, c.W, c.H);
  for (let i = 0; i < f.length; i++) {
    if (Math.abs(f[i] - c.features[i]) > 1e-12) { bad++; console.log('feature', i, c); break; }
  }
  for (let p = 0; p < 2; p++) {
    for (let a = 0; a < pl.A; a++) {
      const rows = c.rows.slice();
      const r = E.drop(rows, c.H, full, pl.cells[p][a]);
      const exp = c.results[p][a];
      if (r !== exp.cleared || (r !== E.GAME_OVER && rows.join() !== exp.rows.join())) {
        bad++; console.log('drop', p, a, r, rows, exp);
      }
    }
  }
}
console.log(bad);
