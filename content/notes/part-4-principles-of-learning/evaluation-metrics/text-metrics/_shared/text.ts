/**
 * Reference implementations of the n-gram and edit-distance text metrics, for the figures in the text-metrics notes.
 * Each follows its primary source; the numbers in the notes were checked against an independent Python version.
 */

export const tokens = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)

function counts<T>(items: T[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const it of items) {
    const k = JSON.stringify(it)
    m.set(k, (m.get(k) ?? 0) + 1)
  }
  return m
}

export const ngrams = (t: string[], n: number): string[][] =>
  Array.from({ length: Math.max(0, t.length - n + 1) }, (_, i) => t.slice(i, i + n))

/** Clipped (modified) n-gram precision: matches are capped by the largest count in any one reference. */
export function modifiedPrecision(hyp: string[], refs: string[][], n: number) {
  const h = counts(ngrams(hyp, n))
  const cap = new Map<string, number>()
  for (const r of refs) for (const [k, c] of counts(ngrams(r, n))) cap.set(k, Math.max(cap.get(k) ?? 0, c))
  let clipped = 0
  let total = 0
  for (const [k, c] of h) {
    clipped += Math.min(c, cap.get(k) ?? 0)
    total += c
  }
  return { clipped, total }
}

/**
 * BLEU (Papineni et al. 2002) over a corpus of (hypothesis, references) pairs, with uniform weights over 1..N-grams.
 * `smooth` adds one to the matched and total counts for n ≥ 2, the smoothing Chen and Cherry call method 2.
 */
export function bleu(pairs: { hyp: string[]; refs: string[][] }[], N = 4, smooth = false) {
  const matched = Array(N).fill(0)
  const total = Array(N).fill(0)
  let c = 0
  let r = 0
  for (const { hyp, refs } of pairs) {
    for (let n = 1; n <= N; n++) {
      const p = modifiedPrecision(hyp, refs, n)
      matched[n - 1] += p.clipped
      total[n - 1] += p.total
    }
    c += hyp.length
    // Effective reference length: the reference length closest to the hypothesis length (shorter wins ties).
    r += refs
      .map((x) => x.length)
      .reduce((best, len) => {
        const d = Math.abs(len - hyp.length)
        const bd = Math.abs(best - hyp.length)
        return d < bd || (d === bd && len < best) ? len : best
      })
  }
  const precisions = matched.map((m, i) =>
    smooth && i > 0 ? (m + 1) / (total[i] + 1) : total[i] > 0 ? m / total[i] : 0,
  )
  const bp = c === 0 ? 0 : c > r ? 1 : Math.exp(1 - r / c)
  const score = precisions.some((p) => p === 0) ? 0 : bp * Math.exp(precisions.reduce((s, p) => s + Math.log(p), 0) / N)
  return { score, precisions, matched, total, bp, hypLength: c, refLength: r }
}

/** chrF (Popović 2015): character n-gram precision and recall averaged over n = 1..N, spaces removed, F_β. */
export function chrf(hyp: string, ref: string, N = 6, beta = 2) {
  const strip = (s: string) => s.replace(/\s+/g, '')
  const h = strip(hyp)
  const r = strip(ref)
  const precisions: number[] = []
  const recalls: number[] = []
  for (let n = 1; n <= N; n++) {
    const hc = counts(Array.from({ length: Math.max(0, h.length - n + 1) }, (_, i) => h.slice(i, i + n)))
    const rc = counts(Array.from({ length: Math.max(0, r.length - n + 1) }, (_, i) => r.slice(i, i + n)))
    let m = 0
    for (const [k, v] of hc) m += Math.min(v, rc.get(k) ?? 0)
    const ht = [...hc.values()].reduce((a, b) => a + b, 0)
    const rt = [...rc.values()].reduce((a, b) => a + b, 0)
    precisions.push(ht ? m / ht : 0)
    recalls.push(rt ? m / rt : 0)
  }
  const P = precisions.reduce((a, b) => a + b, 0) / N
  const R = recalls.reduce((a, b) => a + b, 0) / N
  const b2 = beta * beta
  const F = P + R === 0 ? 0 : ((1 + b2) * P * R) / (b2 * P + R)
  return { F, P, R, precisions, recalls }
}

const f1 = (p: number, r: number) => (p + r === 0 ? 0 : (2 * p * r) / (p + r))

/** ROUGE-N (Lin 2004): n-gram overlap as recall, precision and F1. */
export function rougeN(hyp: string[], ref: string[], n: number) {
  const hc = counts(ngrams(hyp, n))
  const rc = counts(ngrams(ref, n))
  let m = 0
  for (const [k, v] of hc) m += Math.min(v, rc.get(k) ?? 0)
  const P = hc.size ? m / ngrams(hyp, n).length : 0
  const R = rc.size ? m / ngrams(ref, n).length : 0
  return { P, R, F: f1(P, R), matched: m }
}

/** Length of the longest common subsequence of two token lists. */
export function lcs(a: string[], b: string[]): number {
  const d = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0))
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++)
      d[i + 1][j + 1] = a[i] === b[j] ? d[i][j] + 1 : Math.max(d[i][j + 1], d[i + 1][j])
  return d[a.length][b.length]
}

/** ROUGE-L: LCS-based precision, recall and F1. */
export function rougeL(hyp: string[], ref: string[]) {
  const l = lcs(hyp, ref)
  const P = hyp.length ? l / hyp.length : 0
  const R = ref.length ? l / ref.length : 0
  return { P, R, F: f1(P, R), lcs: l }
}

export type EditOp = { op: 'hit' | 'sub' | 'del' | 'ins'; ref?: string; hyp?: string }

/** Minimum-edit (Levenshtein) alignment of reference and hypothesis tokens, with the operations in order. */
export function align(ref: string[], hyp: string[]) {
  const n = ref.length
  const m = hyp.length
  const d = Array.from({ length: n + 1 }, (_, i) =>
    Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
  )
  for (let i = 1; i <= n; i++)
    for (let j = 1; j <= m; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1))
  const ops: EditOp[] = []
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) {
      ops.push({ op: ref[i - 1] === hyp[j - 1] ? 'hit' : 'sub', ref: ref[i - 1], hyp: hyp[j - 1] })
      i--
      j--
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      ops.push({ op: 'del', ref: ref[i - 1] })
      i--
    } else {
      ops.push({ op: 'ins', hyp: hyp[j - 1] })
      j--
    }
  }
  return { distance: d[n][m], ops: ops.reverse() }
}

/** WER and its relatives from an alignment: H hits, S substitutions, D deletions, I insertions. */
export function errorRates(ref: string[], hyp: string[]) {
  const { ops, distance } = align(ref, hyp)
  const H = ops.filter((o) => o.op === 'hit').length
  const S = ops.filter((o) => o.op === 'sub').length
  const D = ops.filter((o) => o.op === 'del').length
  const I = ops.filter((o) => o.op === 'ins').length
  const N = ref.length
  const P = hyp.length
  return {
    ops,
    distance,
    H,
    S,
    D,
    I,
    wer: N ? (S + D + I) / N : 0,
    mer: H + S + D + I ? (S + D + I) / (H + S + D + I) : 0,
    wip: N && P ? (H * H) / (N * P) : 0,
    wil: N && P ? 1 - (H * H) / (N * P) : 1,
  }
}
