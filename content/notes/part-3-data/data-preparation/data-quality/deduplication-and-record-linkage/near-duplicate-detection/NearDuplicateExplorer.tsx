import { useMemo, useState } from 'react'
import { choice, Curve, Figure, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const PRESETS = {
  reworded: {
    label: 'reworded paragraph',
    a: 'Large language models require rigorous deduplication to prevent verbatim memorisation and train-test contamination.',
    b: 'Rigorous deduplication is required for large language models to avoid verbatim memorisation and test set contamination.',
  },
  typo: {
    label: 'minor edit / typo',
    a: 'The quick brown fox jumps over the lazy sleeping canine in the park.',
    b: 'The quick brown fox jumped over the lazy sleeping canine in the park.',
  },
  boilerplate: {
    label: 'shared boilerplate',
    a: 'Copyright 2026 Acme Corp. All rights reserved. Terms and conditions apply. User login page.',
    b: 'Copyright 2026 Acme Corp. All rights reserved. Terms and conditions apply. Contact support page.',
  },
  distinct: {
    label: 'distinct topics',
    a: 'Gradient descent computes the Jacobian matrix of partial derivatives to update network weights iteratively.',
    b: 'Convolutional filters preserve translation equivariance across multi-channel image representations.',
  },
} as const

type PresetKey = keyof typeof PRESETS | 'custom'

const MAX_HASHES = 128

function tokenize(text: string): string[] {
  const clean = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .trim()
  return clean ? clean.split(/\s+/) : []
}

function getShingles(text: string, k: number, unit: 'word' | 'char'): Set<string> {
  const set = new Set<string>()
  if (unit === 'word') {
    const tokens = tokenize(text)
    if (tokens.length < k) {
      if (tokens.length > 0) set.add(tokens.join(' '))
      return set
    }
    for (let i = 0; i + k <= tokens.length; i++) {
      set.add(tokens.slice(i, i + k).join(' '))
    }
    return set
  }
  const clean = text.toLowerCase().replace(/\s+/g, ' ').trim()
  if (clean.length < k) {
    if (clean.length > 0) set.add(clean)
    return set
  }
  for (let i = 0; i + k <= clean.length; i++) {
    set.add(clean.slice(i, i + k))
  }
  return set
}

function fnv1a(str: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

function fmix32(h: number): number {
  h ^= h >>> 16
  h = Math.imul(h, 0x85ebca6b)
  h ^= h >>> 13
  h = Math.imul(h, 0xc2b2ae35)
  h ^= h >>> 16
  return h >>> 0
}

function minhashSignature(shingles: Set<string>, salts: number[]): number[] {
  const items = Array.from(shingles).map(fnv1a)
  return salts.map((salt) => {
    let min = 0xffffffff
    for (const h of items) {
      const v = fmix32(h ^ salt)
      if (v < min) min = v
    }
    return min
  })
}

/** SimHash: 64-bit fingerprint from token feature hashes. */
function simhash64(tokens: string[]): bigint {
  const v = new Array(64).fill(0)
  for (const t of tokens) {
    const h1 = fnv1a(t)
    const h2 = fmix32(h1)
    const combined = (BigInt(h1) << 32n) | BigInt(h2)
    for (let i = 0; i < 64; i++) {
      const bit = (combined >> BigInt(i)) & 1n
      v[i] += bit === 1n ? 1 : -1
    }
  }
  let fingerprint = 0n
  for (let i = 0; i < 64; i++) {
    if (v[i] > 0) fingerprint |= 1n << BigInt(i)
  }
  return fingerprint
}

function hammingDistance64(a: bigint, b: bigint): number {
  let x = a ^ b
  let count = 0
  while (x > 0n) {
    count += Number(x & 1n)
    x >>= 1n
  }
  return count
}

export function NearDuplicateExplorer() {
  const state = useFigureState({
    preset: choice<PresetKey>(
      [
        ...(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((key) => ({
          value: key,
          label: PRESETS[key].label,
        })),
        { value: 'custom', label: 'custom' },
      ],
      'reworded',
      { label: 'example' },
    ),
    method: choice<'minhash' | 'simhash'>(
      [
        { value: 'minhash', label: 'MinHash (shingle Jaccard)' },
        { value: 'simhash', label: 'SimHash (token cosine)' },
      ],
      'minhash',
      { label: 'method' },
    ),
    unit: choice<'word' | 'char'>(
      [
        { value: 'word', label: 'words' },
        { value: 'char', label: 'characters' },
      ],
      'word',
      { label: 'shingle unit' },
    ),
    k: int(3, { min: 1, max: 8, step: 1, label: 'shingle length k' }),
    K: int(64, { min: 4, max: MAX_HASHES, step: 4, label: 'hashes / bits K' }),
    threshold: slider(0.1, 0.9, 0.6, { step: 0.02, label: 'duplicate threshold' }),
    seed: int(42, { ge: 0, label: 'seed' }),
  })

  const [customText] = useState({ a: PRESETS.reworded.a, b: PRESETS.reworded.b })
  const { a, b } = state.preset === 'custom' ? customText : PRESETS[state.preset]

  const salts = useMemo(() => {
    const rng = stream(state.seed)
    return Array.from({ length: MAX_HASHES }, () => Math.floor(uniform(rng) * 0xffffffff) >>> 0)
  }, [state.seed])

  const stats = useMemo(() => {
    const sa = getShingles(a, state.k, state.unit)
    const sb = getShingles(b, state.k, state.unit)
    let intersection = 0
    for (const s of sa) if (sb.has(s)) intersection++
    const union = sa.size + sb.size - intersection
    const jaccard = union === 0 ? 1 : intersection / union

    const sigA = minhashSignature(sa, salts.slice(0, state.K))
    const sigB = minhashSignature(sb, salts.slice(0, state.K))
    let agreeMinhash = 0
    for (let i = 0; i < state.K; i++) {
      if (sigA[i] === sigB[i]) agreeMinhash++
    }
    const minhashEstimate = state.K > 0 ? agreeMinhash / state.K : 0

    const tokensA = tokenize(a)
    const tokensB = tokenize(b)
    const fpA = simhash64(tokensA)
    const fpB = simhash64(tokensB)
    const hammingDist = hammingDistance64(fpA, fpB)
    const simhashSimilarity = 1 - hammingDist / 64

    return {
      saSize: sa.size,
      sbSize: sb.size,
      intersection,
      union,
      jaccard,
      minhashEstimate,
      hammingDist,
      simhashSimilarity,
    }
  }, [a, b, state.k, state.unit, state.K, salts])

  const activeSimilarity = state.method === 'minhash' ? stats.minhashEstimate : stats.simhashSimilarity
  const isDuplicate = activeSimilarity >= state.threshold

  // LSH S-curves for MinHash (b bands of r rows such that b*r = K)
  const lshCurves = useMemo(() => {
    const r = 4
    const b = Math.max(1, Math.floor(state.K / r))
    const xs = Array.from({ length: 101 }, (_, i) => i / 100)
    // P(collision) = 1 - (1 - s^r)^b
    const ys = xs.map((s) => 1 - Math.pow(1 - Math.pow(s, r), b))
    return { xs, ys, b, r }
  }, [state.K])

  const xAxis = useAxis({ label: 'similarity s', range: [0, 1] })
  const yAxis = useAxis({ label: 'probability of candidate collision', range: [0, 1] })

  return (
    <Figure
      title="Near-duplicate detection with MinHash and SimHash"
      state={state}
      hoverReadout={false}
      caption="Compare two texts using k-shingles and hashing. MinHash estimates Jaccard similarity via the fraction of matching minimum hashes across K permutations. SimHash compresses token vectors into a 64-bit fingerprint where Hamming distance reflects cosine similarity. The S-curve shows the LSH candidate collision probability 1 − (1 − s^r)^b (with r = 4 rows per band). Drag the threshold handle to set the boundary between duplicates and distinct texts."
      readouts={
        <>
          <Readout label="true Jaccard J" value={stats.jaccard.toFixed(3)} />
          <Readout
            label={state.method === 'minhash' ? 'MinHash estimate' : 'SimHash similarity'}
            value={activeSimilarity.toFixed(3)}
          />
          <Readout label="decision" value={isDuplicate ? 'near-duplicate' : 'distinct'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="LSH candidate selection probability">
        <Curve
          name={`LSH S-curve (${lshCurves.b} bands × ${lshCurves.r} rows)`}
          x={lshCurves.xs}
          y={lshCurves.ys}
          slot={0}
        />
        <Curve name="active pair similarity" x={[activeSimilarity, activeSimilarity]} y={[0, 1]} slot={1} dashed />
        <Handle
          kind="x"
          at={state.threshold}
          label="threshold"
          onDrag={(x) => state.set('threshold', Math.max(0.05, Math.min(0.95, x)))}
        />
      </Plot>
    </Figure>
  )
}
