/* SZ-Tetris rules and policies for the page.
 * Same rules, placement order and features as src/tetris.py and
 * src/simulate.py (checked by tests/test_engine_js.py).
 * A board is an array of H row bitmasks, row 0 at the bottom. */
(function (root) {
  'use strict';

  var GAME_OVER = -1;
  var LOSE = -1e-12;
  var PIECES = 'SZ';
  // cells (dx, dy) of each piece, dy grows upwards: [piece][orientation]
  var SHAPES = [
    [[[0, 0], [1, 0], [1, 1], [2, 1]], [[0, 1], [0, 2], [1, 0], [1, 1]]],   // S
    [[[1, 0], [2, 0], [0, 1], [1, 1]], [[0, 0], [0, 1], [1, 1], [1, 2]]]    // Z
  ];

  function placements(W) {
    var cells = [[], []], orient = [], column = [];
    for (var p = 0; p < 2; p++) {
      for (var o = 0; o < 2; o++) {
        var shape = SHAPES[p][o];
        var width = 1 + Math.max.apply(null, shape.map(function (c) { return c[0]; }));
        for (var x = 0; x + width <= W; x++) {
          cells[p].push(shape.map(function (c) { return [x + c[0], c[1]]; }));
          if (p === 0) { orient.push(o); column.push(x); }
        }
      }
    }
    return { cells: cells, orient: orient, column: column, A: cells[0].length };
  }

  function fits(rows, H, cells, y) {
    for (var k = 0; k < 4; k++) {
      var r = y + cells[k][1];
      if (r < H && (rows[r] >> cells[k][0]) & 1) return false;
    }
    return true;
  }

  function landingRow(rows, H, cells) {
    var y = H;
    while (y > 0 && fits(rows, H, cells, y - 1)) y--;
    return y;
  }

  /* Drop in place; returns rows cleared or GAME_OVER (board unchanged). */
  function drop(rows, H, full, cells) {
    var y = landingRow(rows, H, cells), k, r;
    for (k = 0; k < 4; k++) if (y + cells[k][1] >= H) return GAME_OVER;
    for (k = 0; k < 4; k++) rows[y + cells[k][1]] |= 1 << cells[k][0];
    var kept = 0;
    for (r = 0; r < H; r++) if (rows[r] !== full) rows[kept++] = rows[r];
    for (r = kept; r < H; r++) rows[r] = 0;
    return H - kept;
  }

  /* Full rows after placing (before clearing), for the animation. */
  function fullRowsAfter(rows, H, full, cells) {
    var y = landingRow(rows, H, cells), tmp = rows.slice(), out = [];
    for (var k = 0; k < 4; k++) {
      if (y + cells[k][1] >= H) return { y: y, over: true, full: [] };
      tmp[y + cells[k][1]] |= 1 << cells[k][0];
    }
    for (var r = 0; r < H; r++) if (tmp[r] === full) out.push(r);
    return { y: y, over: false, full: out };
  }

  /* Packed key, as tetris.pack (exact up to 2^53). */
  function key(rows, W) {
    var k = 0, m = 1, s = Math.pow(2, W);
    for (var r = 0; r < rows.length; r++) { k += rows[r] * m; m *= s; }
    return k;
  }

  function nFeatures(W) { return 2 * W + 2; }

  function features(rows, W, H) {
    var out = new Float64Array(nFeatures(W)), hmax = 0, holes = 0, c, r;
    for (c = 0; c < W; c++) {
      var h = 0;
      for (r = H - 1; r >= 0; r--) if ((rows[r] >> c) & 1) { h = r + 1; break; }
      for (r = 0; r < h; r++) if (!((rows[r] >> c) & 1)) holes++;
      out[c] = h / H;
      if (h > hmax) hmax = h;
    }
    for (c = 0; c < W - 1; c++) out[W + c] = Math.abs(out[c] - out[c + 1]);
    out[2 * W - 1] = hmax / H;
    out[2 * W] = holes / H;
    out[2 * W + 1] = 1;
    return out;
  }

  function dot(x, y) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }

  /* Every placement of piece p: game over flag, rows cleared, next board. */
  function afterstates(rows, W, H, pl, p) {
    var full = (1 << W) - 1, out = [];
    for (var a = 0; a < pl.A; a++) {
      var tmp = rows.slice();
      var c = drop(tmp, H, full, pl.cells[p][a]);
      out.push(c === GAME_OVER ? { over: true, r: 0, rows: null } : { over: false, r: c, rows: tmp });
    }
    return out;
  }

  // ------------------------------------------------------------ random numbers
  function mulberry32(seed) {
    var t = seed >>> 0;
    return function () {
      t = (t + 0x6D2B79F5) >>> 0;
      var x = t;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  // -------------------------------------------------------- decoding the data
  function b64bytes(s) {
    if (typeof atob === 'function') {
      var bin = atob(s), out = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, 'base64'));
  }

  function f16(bits) {
    var s = bits & 0x8000 ? -1 : 1, e = (bits >> 10) & 0x1f, f = bits & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeF16(s) {
    var b = b64bytes(s), n = b.length / 2, out = new Float32Array(n);
    for (var i = 0; i < n; i++) out[i] = f16(b[2 * i] | (b[2 * i + 1] << 8));
    return out;
  }

  /* Table of a small board: sorted keys and lookups. */
  function Table(d) {
    var lo = b64bytes(d.keysLo), hi = b64bytes(d.keysHi), n = d.n;
    var keys = new Float64Array(n);
    var lo32 = new Uint32Array(lo.buffer, lo.byteOffset, n);
    for (var i = 0; i < n; i++) keys[i] = hi[i] * 4294967296 + lo32[i];
    this.keys = keys;
    this.n = n;
    this.wstar = decodeF16(d.wstar);
    this.opt = d.opt;
  }
  Table.prototype.index = function (k) {
    var lo = 0, hi = this.n - 1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1, v = this.keys[mid];
      if (v === k) return mid;
      if (v < k) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  };

  // -------------------------------------------------------------- policies
  /* choose(ctx) -> { a, own: values of the method or null, probs or null }
   * ctx: { rows, W, H, p, pl, after (afterstates), idx (board index or -1),
   *        table, rng (policy random numbers) } */
  function argmaxFirst(q) {
    var best = 0;
    for (var a = 1; a < q.length; a++) if (q[a] > q[best]) best = a;
    return best;
  }

  function tieRandom(q, rng) {
    var best = Math.max.apply(null, q), idx = [];
    for (var a = 0; a < q.length; a++) if (q[a] === best) idx.push(a);
    return idx[Math.floor(rng() * idx.length)];
  }

  var policies = {
    random: function (ctx) {
      return { a: Math.floor(ctx.rng() * ctx.pl.A), own: null, probs: null };
    },
    greedy: function (ctx) {
      var q = ctx.after.map(function (s) { return s.over ? -1000 : s.r; });
      return { a: tieRandom(q, ctx.rng), own: ctx.after.map(function (s) { return s.over ? null : s.r; }), probs: null };
    },
    table: function (ctx, actions) {
      // a learned or optimal policy stored as one letter per (board, piece)
      return { a: actions.charCodeAt(2 * ctx.idx + ctx.p) - 97, own: null, probs: null };
    },
    afterstateTable: function (ctx, actions, Wv) {
      var t = ctx.table, own = ctx.after.map(function (s) {
        return s.over ? 0 : s.r + Wv[t.index(key(s.rows, ctx.W))];
      });
      return { a: actions.charCodeAt(2 * ctx.idx + ctx.p) - 97, own: own, probs: null };
    },
    linear: function (ctx, w) {
      var q = ctx.after.map(function (s) {
        return s.over ? LOSE : s.r + dot(features(s.rows, ctx.W, ctx.H), w);
      });
      return { a: argmaxFirst(q), own: q.map(function (v) { return v === LOSE ? 0 : v; }), probs: null };
    },
    softmax: function (ctx, theta) {
      var nf = nFeatures(ctx.W);
      var h = ctx.after.map(function (s) {
        if (s.over) return theta[nf + 1];
        return dot(features(s.rows, ctx.W, ctx.H), theta.slice(0, nf)) + theta[nf] * s.r;
      });
      var m = Math.max.apply(null, h), e = h.map(function (v) { return Math.exp(v - m); });
      var z = e.reduce(function (x, y) { return x + y; }, 0), probs = e.map(function (v) { return v / z; });
      var u = ctx.rng(), c = 0, a = probs.length - 1;
      for (var k = 0; k < probs.length; k++) { c += probs[k]; if (u < c) { a = k; break; } }
      return { a: a, own: null, probs: probs };
    }
  };

  /* Exact values r + W*(b') of every placement (0 if it loses). */
  function exactValues(ctx) {
    var t = ctx.table;
    if (!t) return null;
    return ctx.after.map(function (s) { return s.over ? 0 : s.r + t.wstar[t.index(key(s.rows, ctx.W))]; });
  }

  var Engine = {
    GAME_OVER: GAME_OVER, PIECES: PIECES, placements: placements, landingRow: landingRow,
    drop: drop, fullRowsAfter: fullRowsAfter, key: key, features: features,
    nFeatures: nFeatures, afterstates: afterstates, mulberry32: mulberry32,
    decodeF16: decodeF16, b64bytes: b64bytes, Table: Table, policies: policies,
    exactValues: exactValues, dot: dot
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = Engine;
  else root.Engine = Engine;
})(this);
