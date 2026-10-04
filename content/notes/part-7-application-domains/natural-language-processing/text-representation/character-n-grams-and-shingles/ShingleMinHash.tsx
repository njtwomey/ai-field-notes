import { useMemo, useState } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Readout, Textarea, useAxis, useFigureState } from 'aifn-render'
import { jaccard, minhash, shingles } from '../_shared/text'
import { stream, uniform } from 'aifn/foundation/random'

const PRESETS = {
  rose: { label: 'a rose', a: 'a rose is a rose is a rose', b: 'a rose is a flower which is a rose' },
  fox: {
    label: 'one edit',
    a: 'the quick brown fox jumps over the lazy dog',
    b: 'the quick brown fox jumped over the lazy dog',
  },
  news: {
    label: 'reworded news',
    a: 'The central bank raised interest rates by a quarter point on Tuesday, citing persistent inflation in services.',
    b: 'On Tuesday the central bank raised interest rates by a quarter point, citing persistent services inflation.',
  },
} as const
type Preset = keyof typeof PRESETS | 'custom'

const MAX_HASHES = 256

/** Shingle two texts, compare the exact resemblance with its MinHash estimate as hash functions are added. */
export function ShingleMinHash() {
  const state = useFigureState({
    preset: choice<Preset>(
      [
        ...(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((key) => ({ value: key, label: PRESETS[key].label })),
        { value: 'custom', label: 'custom' },
      ],
      'fox',
      { label: 'example' },
    ),
    unit: choice<'word' | 'char'>(
      [
        { value: 'word', label: 'words' },
        { value: 'char', label: 'characters' },
      ],
      'word',
      { label: 'shingle unit' },
    ),
    k: int(2, { min: 1, max: 10, step: 1, label: 'shingle size k' }),
    K: int(64, { min: 1, max: MAX_HASHES, step: 1, label: 'hash functions K', format: (v) => v.toFixed(0) }),
    seed: int(5, { ge: 0, label: 'seed' }),
  })

  // Typed texts are kept while an example is shown, and come back with "custom".
  const [custom, setCustom] = useState<{ a: string; b: string }>({ a: PRESETS.fox.a, b: PRESETS.fox.b })
  const { a, b } = state.preset === 'custom' ? custom : PRESETS[state.preset]
  const edit = (next: { a: string; b: string }) => {
    setCustom(next)
    state.set('preset', 'custom')
  }

  // Salts for the hash functions: fixed per seed, so moving the K slider only reveals more of the same functions.
  const salts = useMemo(() => {
    const r = stream(state.seed)
    return Array.from({ length: MAX_HASHES }, () => Math.floor(uniform(r) * 0xffffffff) >>> 0)
  }, [state.seed])

  const m = useMemo(() => {
    const sa = shingles(a, state.unit, state.k)
    const sb = shingles(b, state.unit, state.k)
    let inter = 0
    for (const x of sa) if (sb.has(x)) inter++
    const J = jaccard(sa, sb)
    const ha = minhash(sa, salts)
    const hb = minhash(sb, salts)
    // Running estimate: the fraction of agreeing minima among the first j hash functions.
    const est: number[] = []
    let agree = 0
    for (let j = 0; j < MAX_HASHES; j++) {
      if (sa.size && sb.size && ha[j] === hb[j]) agree++
      est.push(agree / (j + 1))
    }
    return { sa, sb, inter, J, est }
  }, [a, b, state.unit, state.k, salts])

  const series = useMemo(() => {
    const xs = Array.from({ length: state.K }, (_, i) => i + 1)
    const sd = xs.map((j) => Math.sqrt((m.J * (1 - m.J)) / j))
    return [
      { name: 'MinHash estimate', x: xs, y: m.est.slice(0, state.K), slot: 0 },
      { name: 'true resemblance J', x: [1, state.K], y: [m.J, m.J], muted: true, dashed: true },
      {
        name: 'J ± 2 standard deviations',
        x: xs,
        y: sd.map((s) => Math.min(1, m.J + 2 * s)),
        slot: 1,
        dashed: true,
      },
      { name: 'J − 2 sd', x: xs, y: sd.map((s) => Math.max(0, m.J - 2 * s)), slot: 1, dashed: true },
    ] as const
  }, [m, state.K])

  const estimate = m.est[state.K - 1]
  const xAxis = useAxis({ label: 'number of hash functions', range: [1, Math.max(2, state.K)] })
  const yAxis = useAxis({ label: 'resemblance', range: [0, 1] })
  return (
    <Figure
      title="Shingles and MinHash"
      state={state}
      caption="Edit the two texts or pick an example. Each text becomes its set of k-shingles (runs of k words or k characters). The resemblance J is the Jaccard similarity of the two sets. MinHash keeps, for each hash function, the smallest hash in each set; the fraction of functions whose minima agree estimates J. The estimate wanders early and settles inside the ±2 standard-deviation band √(J(1 − J)/K) as functions are added."
      readouts={
        <>
          <Readout label="|S(A)|" value={m.sa.size} />
          <Readout label="|S(B)|" value={m.sb.size} />
          <Readout label="shared" value={m.inter} />
          <Readout label="resemblance J" value={formatNumber(m.J)} />
          <Readout label="containment of A in B" value={formatNumber(m.sa.size ? m.inter / m.sa.size : 0)} />
          <Readout label={`MinHash estimate, K = ${state.K}`} value={formatNumber(estimate)} />
          <Readout label="standard deviation" value={formatNumber(Math.sqrt((m.J * (1 - m.J)) / state.K))} />
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs text-muted-foreground">
          text A
          <Textarea
            value={a}
            onChange={(e) => {
              edit({ a: e.target.value, b })
            }}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
        <label className="space-y-1 text-xs text-muted-foreground">
          text B
          <Textarea
            value={b}
            onChange={(e) => {
              edit({ a, b: e.target.value })
            }}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
      </div>
      <Plot x={xAxis} y={yAxis} height={260}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
