/*
 * hand_core.js  -  three.js 非依存の計算コア
 *   1. GLB パーサ (build_hands.py の出力専用)
 *   2. 骨格 (build_hands.py の build_finger の移植)
 *   3. スキンウェイトの自動生成 (読み込み時に1回だけ)
 *   4. MediaPipe のランドマーク -> 16本のボーン行列 (毎フレーム、ほぼ無料)
 *
 * ボーン番号: 0 = 手のひら(手首含む) / 1 + 3*f + k = 指 f (0=thumb..4=pinky) の第 k 節
 * 単位は m。メッシュの座標系: 指先 +Y / 手の甲 +Z / 右手の親指は -X。
 */
(function (root) {
  'use strict';

  const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  const NBONES = 16;

  // ---- build_hands.py と同じ基準形状 (cm) ----
  const FINGER_BASE = {
    index:  { base: [-0.86, 7.30, -0.1], L: [3.11, 2.03, 1.62], R: [0.84, 0.78, 0.72, 0.6],  splay: 0.04, cf: 0.80 },
    middle: { base: [0.83, 6.95, -0.1],  L: [3.83, 2.5, 2.0],   R: [0.86, 0.8, 0.73, 0.6],   splay: 0.23, cf: 1.00 },
    ring:   { base: [2.33, 6.45, -0.1],  L: [3.52, 2.29, 1.84], R: [0.8, 0.74, 0.68, 0.57],  splay: 0.44, cf: 1.15 },
    pinky:  { base: [3.25, 5.93, -0.1],  L: [2.71, 1.77, 1.41], R: [0.72, 0.66, 0.6, 0.5],   splay: 0.87, cf: 1.30 },
  };
  const THUMB = { base: [-1.90, 1.20, -0.4], L: [3.90, 2.40, 1.50], R: [1.2, 1.08, 0.94, 0.74],
                  yaw: [42, 50, 42], flex: [5, 3, 3] };
  const FLEX_FULL = [25, 40, 25];

  const DEFAULT_PARAMS = {
    hand_scale: 1.0, palm_width: 0.92, palm_length: 0.88, length_scale: 1.2, thickness_scale: 0.82,
    curl: 0.04, spread: 1.0, thumb_spread: 1.0, shade_strength: 0.4, outline_width_cm: 0.035,
    skin_color: '#FBECE4', outline_color: '#0d0d0d',
    fingers: {
      thumb: { length: 0.88, thickness: 0.85 }, index: { length: 1, thickness: 1 },
      middle: { length: 1, thickness: 1 }, ring: { length: 1, thickness: 1 }, pinky: { length: 1, thickness: 1 },
    },
  };

  // ---- 小さなベクトル/行列ユーティリティ (配列, 行優先3x3) ----
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const smooth = (e0, e1, x) => { const t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); };
  const matVec = (M, v) => [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[3] * v[0] + M[4] * v[1] + M[5] * v[2], M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];
  function matMul(A, B) {
    const C = new Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    return C;
  }
  function rotBetween(u, w) {            // u -> w の最小回転
    const c = dot(u, w);
    if (c < -0.99999) {
      const ax = norm(cross(u, Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
      return [2 * ax[0] * ax[0] - 1, 2 * ax[0] * ax[1], 2 * ax[0] * ax[2],
              2 * ax[1] * ax[0], 2 * ax[1] * ax[1] - 1, 2 * ax[1] * ax[2],
              2 * ax[2] * ax[0], 2 * ax[2] * ax[1], 2 * ax[2] * ax[2] - 1];
    }
    const v = cross(u, w), k = 1 / (1 + c);
    const vx = [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0];
    const vx2 = matMul(vx, vx);
    const R = new Array(9);
    for (let i = 0; i < 9; i++) R[i] = (i % 4 === 0 ? 1 : 0) + vx[i] + k * vx2[i];
    return R;
  }
  function eulerMat(rx, ry, rz) {
    const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
    const X = [1, 0, 0, 0, cx, -sx, 0, sx, cx], Y = [cy, 0, sy, 0, 1, 0, -sy, 0, cy], Z = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
    return matMul(Z, matMul(Y, X));
  }

  // ---------------------------------------------------------------------------
  // GLB パーサ
  // ---------------------------------------------------------------------------
  function parseGLB(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('GLBではありません');
    const jl = dv.getUint32(12, true);
    const gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jl)));
    const binOff = 20 + jl + 8;
    const view = (acc, Type, comps) => {
      const a = gltf.accessors[acc], bv = gltf.bufferViews[a.bufferView];
      const off = binOff + (bv.byteOffset || 0) + (a.byteOffset || 0);
      return new Type(buf.slice(off, off + a.count * comps * Type.BYTES_PER_ELEMENT));
    };
    const params = (gltf.asset && gltf.asset.extras && gltf.asset.extras.params) || null;
    const hands = {};
    gltf.meshes.forEach((m) => {
      const side = /_L$/.test(m.name) ? 'L' : 'R';
      const skin = m.primitives[0];
      const outl = m.primitives.find((p, i) => i > 0 && !p.attributes.NORMAL);
      const h = {
        pos: view(skin.attributes.POSITION, Float32Array, 3),
        nor: view(skin.attributes.NORMAL, Float32Array, 3),
        col: skin.attributes.COLOR_0 !== undefined ? view(skin.attributes.COLOR_0, Float32Array, 3) : null,
        idx: view(skin.indices, Uint32Array, 1),
        ow: null,
      };
      const n = h.pos.length / 3;
      h.ow = new Float32Array(n);
      if (outl) {                                  // 輪郭用メッシュ = 頂点を法線方向に押し出したもの。幅だけ取り出す
        const ov = view(outl.attributes.POSITION, Float32Array, 3);
        for (let i = 0; i < n; i++) h.ow[i] = Math.hypot(ov[3 * i] - h.pos[3 * i], ov[3 * i + 1] - h.pos[3 * i + 1], ov[3 * i + 2] - h.pos[3 * i + 2]);
      } else h.ow.fill(0.00035);
      if (!h.col) { h.col = new Float32Array(n * 3).fill(1); }
      hands[side] = h;
    });
    return { params, hands };
  }

  // ---------------------------------------------------------------------------
  // 骨格 (build_hands.py: build_finger の移植)。メートルに変換し、左手なら X を反転
  // ---------------------------------------------------------------------------
  function buildSkeleton(userParams, left) {
    const P = Object.assign({}, DEFAULT_PARAMS, userParams || {});
    P.fingers = Object.assign({}, DEFAULT_PARAMS.fingers, (userParams && userParams.fingers) || {});
    const k = 0.01 * P.hand_scale, deg = Math.PI / 180;
    const fx = left ? -1 : 1;
    const fingers = FINGERS.map((name) => {
      const f = P.fingers[name];
      const ls = Array.isArray(f.length) ? f.length : [f.length, f.length, f.length];
      const thick = f.thickness * P.thickness_scale;
      let spec, L, splays, flexCum;
      const cum = (a) => { let s = 0; return a.map((x) => (s += x)); };
      if (name === 'thumb') {
        spec = THUMB;
        L = [0, 1, 2].map((i) => spec.L[i] * ls[i] * P.length_scale);
        splays = spec.yaw.map((y) => -y * P.thumb_spread * deg);
        flexCum = cum(spec.flex.map((a) => (a + 20 * P.curl) * deg));
      } else {
        spec = FINGER_BASE[name];
        L = [0, 1, 2].map((i) => spec.L[i] * ls[i] * P.length_scale);
        const sp = spec.splay * P.spread;
        splays = [sp, sp, sp];
        flexCum = cum(FLEX_FULL.map((a) => a * P.curl * spec.cf * deg));
      }
      const base = [spec.base[0] * P.palm_width, spec.base[1] * P.palm_length, spec.base[2]];
      const R = spec.R.map((x) => x * thick);
      const pts = [base], dirs = [];
      for (let i = 0; i < 3; i++) {
        const d = norm([Math.sin(splays[i]) * Math.cos(flexCum[i]), Math.cos(splays[i]) * Math.cos(flexCum[i]), -Math.sin(flexCum[i])]);
        dirs.push(d);
        pts.push(add(pts[i], mul(d, L[i])));
      }
      const m = (v) => [v[0] * k * fx, v[1] * k, v[2] * k];
      const mp = pts.map(m), md = dirs.map((d) => [d[0] * fx, d[1], d[2]]);
      const Lm = [0, 1, 2].map((i) => len(sub(mp[i + 1], mp[i])));
      return { name, pts: mp, dirs: md, R: R.map((x) => x * k), L: Lm, total: Lm[0] + Lm[1] + Lm[2] };
    });
    // 手首 (ランドマーク0 に相当) と、手のひらの基準フレーム
    const wrist = [0.3 * P.palm_width * k * fx, 0.2 * k, 0];
    const bases = [1, 2, 3, 4].map((i) => fingers[i].pts[0]);   // index, middle, ring, pinky
    const frame = frameFrom(wrist, bases[0], bases[1], bases[2], bases[3], left);
    return {
      left, fingers, wrist, frame,
      span: len(sub(bases[0], bases[3])),
      palmLen: len(sub(frame.o, wrist)),
    };
  }

  // 手のひらの座標系。X = 小指側 (左手は親指側), Y = 手首->指の付け根, Z = X x Y = 手の甲側
  function frameFrom(w, i, m, r, p, left) {
    const mc = mul(add(add(i, m), add(r, p)), 0.25);
    const X = norm(left ? sub(i, p) : sub(p, i));
    let Y = sub(mc, w);
    Y = norm(sub(Y, mul(X, dot(Y, X))));
    return { o: mc, B: [X, Y, cross(X, Y)] };
  }
  function rotFromFrames(Bt, Br) {       // Bt の各軸 <- Br の各軸 に移す回転
    const R = new Array(9).fill(0);
    for (let c = 0; c < 3; c++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) R[i * 3 + j] += Bt[c][i] * Br[c][j];
    return R;
  }

  // ---------------------------------------------------------------------------
  // スキンウェイト自動生成。各頂点を「いちばん近い指の軸 (カプセル)」に割り当て、
  // 指の軸方向の位置 s から 手のひら / 第1 / 第2 / 第3 節 へ滑らかに配分する。
  // 戻り値: skinIndex(Uint16 x4), skinWeight(Float32 x4)。スロット順は [palm, seg0, seg1, seg2]
  // ---------------------------------------------------------------------------
  function computeWeights(pos, skel) {
    const n = pos.length / 3;
    const idx = new Uint16Array(n * 4), wt = new Float32Array(n * 4), rc = new Float32Array(n * 4);
    const F = skel.fingers;
    const ratio = new Float64Array(5), sArr = new Float64Array(5), cp = new Float64Array(15);
    const infl = (f) => 1 - smooth(f === 0 ? 1.7 : 1.45, f === 0 ? 2.8 : 2.1, ratio[f]);
    const base = (f) => infl(f) * smooth(-F[f].R[0] * (f === 0 ? 1.6 : 1.2), F[f].R[0] * (f === 0 ? 1.6 : 1.2), sArr[f]);
    for (let v = 0; v < n; v++) {
      const px = pos[3 * v], py = pos[3 * v + 1], pz = pos[3 * v + 2];
      for (let f = 0; f < 5; f++) {                      // 指ごとに、軸 (3節のカプセル) までの最短を求める
        const fg = F[f];
        let best = 1e9, bs = 0, cumL = 0;
        for (let k = 0; k < 3; k++) {
          const a = fg.pts[k], b = fg.pts[k + 1];
          const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
          const t = ((px - a[0]) * bx + (py - a[1]) * by + (pz - a[2]) * bz) / (fg.L[k] * fg.L[k]);
          const tc = t < 0 ? 0 : t > 1 ? 1 : t;
          const qx = a[0] + bx * tc, qy = a[1] + by * tc, qz = a[2] + bz * tc;
          const dx = px - qx, dy = py - qy, dz = pz - qz;
          const r = fg.R[k] + (fg.R[k + 1] - fg.R[k]) * tc;
          const ra = Math.sqrt(dx * dx + dy * dy + dz * dz) / r;
          if (ra < best) { best = ra; bs = cumL + t * fg.L[k]; cp[3 * f] = qx; cp[3 * f + 1] = qy; cp[3 * f + 2] = qz; }
          cumL += fg.L[k];
        }
        ratio[f] = best; sArr[f] = bs;
      }
      let A = 0; for (let f = 1; f < 5; f++) if (ratio[f] < ratio[A]) A = f;
      let B = -1; for (let f = 0; f < 5; f++) if (f !== A && (B < 0 || ratio[f] < ratio[B])) B = f;
      const fa = F[A], fb = F[B];
      const FA = base(A), FB = base(B);
      const j1 = fa.L[0], j2 = fa.L[0] + fa.L[1];
      const wA = smooth(j1 - fa.R[1], j1 + fa.R[1], sArr[A]);
      const wB = smooth(j2 - fa.R[2] * 0.9, j2 + fa.R[2] * 0.9, sArr[A]);
      // 付け根付近の「指と指の間」は、両側の指の第1節で滑らかに分け合う (片方に固定すると曲げたときに裂ける)
      const shareA = smooth(-0.5, 0.5, ratio[B] - ratio[A]);
      let sb = (1 - shareA) * (1 - smooth(0.35 * j1, 0.7 * j1, sArr[A])) * (1 - smooth(0.35 * fb.L[0], 0.7 * fb.L[0], sArr[B]));
      if (wA > 1e-4) sb = 0;
      const cA = FA * (1 - sb), cB = FB * sb;
      const o = 4 * v, a0 = 1 + 3 * A, b0 = 1 + 3 * B;
      idx[o] = 0; idx[o + 1] = a0;
      wt[o] = 1 - cA - cB; wt[o + 1] = cA * (1 - wA);
      if (sb > 0) { idx[o + 2] = b0; idx[o + 3] = a0 + 2; wt[o + 2] = cB; wt[o + 3] = 0; }
      else { idx[o + 2] = a0 + 1; idx[o + 3] = a0 + 2; wt[o + 2] = cA * (wA - wB); wt[o + 3] = cA * wB; }
      // 体積補正用: 軸から頂点へのベクトル (休止姿勢) と、補正の強さ
      // 補正は指の胴体だけに効かせる。付け根の手前・指と指の間 (隣の指の軸に切り替わる場所) では 0 にして、継ぎ目を作らない
      const thumbA = A === 0;
      const body = 1 - smooth(thumbA ? 1.5 : 1.15, thumbA ? 2.2 : 1.5, ratio[A]);
      const kk = body * smooth(0, 0.8 * fa.R[0], sArr[A]) * (1 - 2 * sb);
      rc[o] = px - cp[3 * A]; rc[o + 1] = py - cp[3 * A + 1]; rc[o + 2] = pz - cp[3 * A + 2]; rc[o + 3] = kk;
    }
    return { idx, wt, rc };
  }

  // ---------------------------------------------------------------------------
  // MediaPipe の出力 -> カメラ空間 (m, three.js と同じ向き: +X 右 / +Y 上 / -Z 奥)
  //   img: 正規化画像座標 [{x,y}], wld: ワールドランドマーク [{x,y,z}] (m, 手の中心原点)
  //   奥行きは「手のひら周りの実寸 / 画像上の長さ」から推定する
  // ---------------------------------------------------------------------------
  function camSpace(img, wld, o) {
    const ty = Math.tan(o.fovY * Math.PI / 360), mx = o.mirror ? -1 : 1;
    const segs = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17], [5, 17]];
    let sw = 0, sp = 0;
    for (const [a, b] of segs) {
      sw += Math.hypot(wld[a].x - wld[b].x, wld[a].y - wld[b].y);
      sp += Math.hypot((img[a].x - img[b].x) * o.aspect * 2 * ty, (img[a].y - img[b].y) * 2 * ty);
    }
    const z = Math.min(2.5, Math.max(0.12, sw / Math.max(sp, 1e-6)));
    const set = [0, 5, 9, 13, 17];
    let cx = 0, cy = 0, wx = 0, wy = 0, wz = 0;
    for (const i of set) { cx += img[i].x; cy += img[i].y; wx += wld[i].x; wy += wld[i].y; wz += wld[i].z; }
    cx /= 5; cy /= 5; wx /= 5; wy /= 5; wz /= 5;
    const C = [mx * (2 * cx - 1) * o.aspect * ty * z, (1 - 2 * cy) * ty * z, -z];
    const P = new Float64Array(63);
    for (let i = 0; i < 21; i++) {
      P[3 * i] = C[0] + mx * (wld[i].x - wx);
      P[3 * i + 1] = C[1] - (wld[i].y - wy);
      P[3 * i + 2] = C[2] - (wld[i].z - wz);
    }
    return P;
  }

  // ---------------------------------------------------------------------------
  // ポーズ -> ボーン行列。P はカメラ空間の 21 点 (Float64Array(63))
  //   o = { size, len, th, fLen[5], fTh[5], fit }   st = 呼び出し間で持ち回る状態 ({} でよい)
  //   out の各ボーン: 列優先 4x4。メッシュ(休止姿勢)の頂点にそのまま掛ければ現在の姿勢になる
  // ---------------------------------------------------------------------------
  function put(out, b, M3, t) {
    const o = b * 16;
    for (let r = 0; r < 3; r++) { for (let c = 0; c < 3; c++) out[o + c * 4 + r] = M3[r * 3 + c]; out[o + 12 + r] = t[r]; out[o + 3 + 4 * r] = 0; }
    out[o + 15] = 1;
  }
  function solve(P, skel, o, st, out) {
    out = out || new Float32Array(NBONES * 16);
    const g = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]];
    const Ft = frameFrom(g(0), g(5), g(9), g(13), g(17), skel.left);
    const Rp = rotFromFrames(Ft.B, skel.frame.B);
    const spT = 0.5 * (len(sub(g(5), g(17))) / skel.span + len(sub(Ft.o, g(0))) / skel.palmLen);
    st.sp = st.sp ? st.sp + (spT - st.sp) * 0.1 : spT;
    const sp = st.sp * o.size;
    // 手のひらボーン
    const Mp = Rp.map((x) => x * sp);
    const orr = matVec(Mp, skel.frame.o);
    put(out, 0, Mp, sub(Ft.o, orr));
    st.sf = st.sf || [0, 0, 0, 0, 0];
    for (let f = 0; f < 5; f++) {
      const fg = skel.fingers[f];
      const lm = f === 0 ? 1 : 5 + 4 * (f - 1);          // 付け根のランドマーク番号 (親指は CMC)
      // 指の実寸に合わせた倍率 (ゆっくり追従させて、ランドマークのゆらぎで伸び縮みしないようにする)
      let trk = 0;
      for (let k = 0; k < 3; k++) trk += len(sub(g(lm + k + 1), g(lm + k)));
      const sfT = o.fit ? trk / fg.total * o.size : sp;
      st.sf[f] = st.sf[f] ? st.sf[f] + (sfT - st.sf[f]) * 0.05 : sfT;
      const sf = st.sf[f];
      const lenF = o.len * o.fLen[f], thF = o.th * o.fTh[f];
      let A = add(matVec(Mp, sub(fg.pts[0], skel.frame.o)), Ft.o);
      for (let k = 0; k < 3; k++) {
        const D = norm(sub(g(lm + k + 1), g(lm + k)));
        const d = fg.dirs[k];
        const Rk = matMul(rotBetween(matVec(Rp, d), D), Rp);
        // 休止姿勢での異方スケール: 軸方向に lenF、断面に thF
        // 太さは手のひらと指の倍率の中間 (指の長さに合わせて細くなりすぎないように)
        const tb = Math.sqrt(sp * sf) * thF, a = tb, b = sf * lenF - tb;
        const S = [a + b * d[0] * d[0], b * d[0] * d[1], b * d[0] * d[2],
                   b * d[1] * d[0], a + b * d[1] * d[1], b * d[1] * d[2],
                   b * d[2] * d[0], b * d[2] * d[1], a + b * d[2] * d[2]];
        const M3 = matMul(Rk, S);
        put(out, 1 + 3 * f + k, M3, sub(A, matVec(M3, fg.pts[k])));
        A = add(A, mul(D, fg.L[k] * sf * lenF));
      }
    }
    return { M: out, sp };
  }

  // 検証用の CPU スキニング。シェーダーと同じ式 (rc を渡すと体積補正が入る)
  function applySkin(pos, idx, wt, M, out, rc) {
    const n = pos.length / 3;
    out = out || new Float32Array(pos.length);
    for (let v = 0; v < n; v++) {
      const x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
      const rx = rc ? rc[4 * v] : 0, ry = rc ? rc[4 * v + 1] : 0, rz = rc ? rc[4 * v + 2] : 0, k = rc ? rc[4 * v + 3] : 0;
      let ox = 0, oy = 0, oz = 0, cx = 0, cy = 0, cz = 0, el = 0;
      for (let j = 0; j < 4; j++) {
        const w = wt[4 * v + j]; if (w === 0) continue;
        const m = idx[4 * v + j] * 16;
        ox += w * (M[m] * x + M[m + 4] * y + M[m + 8] * z + M[m + 12]);
        oy += w * (M[m + 1] * x + M[m + 5] * y + M[m + 9] * z + M[m + 13]);
        oz += w * (M[m + 2] * x + M[m + 6] * y + M[m + 10] * z + M[m + 14]);
        if (k > 0) {
          const ax = x - rx, ay = y - ry, az = z - rz;
          cx += w * (M[m] * ax + M[m + 4] * ay + M[m + 8] * az + M[m + 12]);
          cy += w * (M[m + 1] * ax + M[m + 5] * ay + M[m + 9] * az + M[m + 13]);
          cz += w * (M[m + 2] * ax + M[m + 6] * ay + M[m + 10] * az + M[m + 14]);
          el += w * Math.hypot(M[m] * rx + M[m + 4] * ry + M[m + 8] * rz, M[m + 1] * rx + M[m + 5] * ry + M[m + 9] * rz, M[m + 2] * rx + M[m + 6] * ry + M[m + 10] * rz);
        }
      }
      if (k > 0) {
        const dx = ox - cx, dy = oy - cy, dz = oz - cz, f = Math.min(el / Math.max(Math.hypot(dx, dy, dz), 1e-7), 2.2);
        ox += (cx + dx * f - ox) * k; oy += (cy + dy * f - oy) * k; oz += (cz + dz * f - oz) * k;
      }
      out[3 * v] = ox; out[3 * v + 1] = oy; out[3 * v + 2] = oz;
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // デモ用の合成ポーズ (カメラなしで確認するため)。curl は指ごとに 0(開)〜1(握り)
  // ---------------------------------------------------------------------------
  function synthHand(skel, curl, g) {
    const pts = new Array(21);
    pts[0] = skel.wrist.slice();
    const ang = [85, 100, 60].map((a) => a * Math.PI / 180);
    for (let f = 0; f < 5; f++) {
      const fg = skel.fingers[f], lm = f === 0 ? 1 : 5 + 4 * (f - 1);
      const c = curl[f], th = f === 0;
      let p = fg.pts[0].slice(), phi = 0;
      pts[lm] = p;
      for (let k = 0; k < 3; k++) {
        phi += c * ang[k] * (th ? [0.3, 0.55, 0.6][k] : 1);
        const d = fg.dirs[k];
        const dn = [d[0], d[1] * Math.cos(phi) + d[2] * Math.sin(phi), -d[1] * Math.sin(phi) + d[2] * Math.cos(phi)];
        p = add(p, mul(dn, fg.L[k]));
        pts[lm + k + 1] = p;
      }
    }
    const R = eulerMat(g.rx, g.ry, g.rz), P = new Float64Array(63);
    for (let i = 0; i < 21; i++) {
      const q = add(matVec(R, mul(sub(pts[i], skel.frame.o), g.scale)), g.pos);
      P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2];
    }
    return P;
  }

  // ---------------------------------------------------------------------------
  // One Euro フィルタ (ランドマークのブレ取り。速い動きには遅れず、止まっているときは安定)
  // ---------------------------------------------------------------------------
  function OneEuro(n, minCut, beta) {
    this.n = n; this.minCut = minCut || 1.5; this.beta = beta || 10; this.dCut = 1;
    this.x = null; this.dx = new Float64Array(n);
  }
  OneEuro.prototype.reset = function () { this.x = null; };
  OneEuro.prototype.filter = function (v, t) {          // v: Float64Array(n) を書き換える。t: 秒
    if (!this.x) { this.x = Float64Array.from(v); this.t = t; this.dx.fill(0); return v; }
    const dt = Math.max(t - this.t, 1e-3); this.t = t;
    const al = (c) => 1 / (1 + 1 / (2 * Math.PI * c) / dt);
    const ad = al(this.dCut);
    for (let i = 0; i < this.n; i++) {
      const d = (v[i] - this.x[i]) / dt;
      this.dx[i] += (d - this.dx[i]) * ad;
      const a = al(this.minCut + this.beta * Math.abs(this.dx[i]));
      this.x[i] += (v[i] - this.x[i]) * a;
      v[i] = this.x[i];
    }
    return v;
  };

  const api = { FINGERS, NBONES, DEFAULT_PARAMS, parseGLB, buildSkeleton, computeWeights, camSpace, solve, applySkin, synthHand, OneEuro };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.HandCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
