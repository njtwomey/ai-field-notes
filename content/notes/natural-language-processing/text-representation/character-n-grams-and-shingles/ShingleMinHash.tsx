import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from '@/components/viz'
import { Textarea } from '@/components/ui/textarea'
import { rng } from '@/lib/math'
import { jaccard, minhash, shingles } from '../_shared/text'

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
  const [preset, setPreset] = useState<Preset>('fox')
  const [a, setA] = useState<string>(PRESETS.fox.a)
  const [b, setB] = useState<string>(PRESETS.fox.b)
  const [unit, setUnit] = useState<'word' | 'char'>('word')
  const [k, setK] = useState(2)
  const [K, setKHashes] = useState(64)
  const [seed, setSeed] = useState(5)

  const choose = (p: Preset) => {
    setPreset(p)
    if (p !== 'custom') {
      setA(PRESETS[p].a)
      setB(PRESETS[p].b)
    }
  }

  // Salts for the hash functions: fixed per seed, so moving the K slider only reveals more of the same functions.
  const salts = useMemo(() => {
    const r = rng(seed)
    return Array.from({ length: MAX_HASHES }, () => Math.floor(r.uniform() * 0xffffffff) >>> 0)
  }, [seed])

  const m = useMemo(() => {
    const sa = shingles(a, unit, k)
    const sb = shingles(b, unit, k)
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
  }, [a, b, unit, k, salts])

  const series = useMemo<XYSeries[]>(() => {
    const xs = Array.from({ length: K }, (_, i) => i + 1)
    const sd = xs.map((j) => Math.sqrt((m.J * (1 - m.J)) / j))
    return [
      { name: 'MinHash estimate', type: 'line', x: xs, y: m.est.slice(0, K), slot: 0 },
      { name: 'true resemblance J', type: 'line', x: [1, K], y: [m.J, m.J], muted: true, dashed: true },
      {
        name: 'J ± 2 standard deviations',
        type: 'line',
        x: xs,
        y: sd.map((s) => Math.min(1, m.J + 2 * s)),
        slot: 1,
        dashed: true,
      },
      { name: 'J − 2 sd', type: 'line', x: xs, y: sd.map((s) => Math.max(0, m.J - 2 * s)), slot: 1, dashed: true },
    ]
  }, [m, K])

  const estimate = m.est[K - 1]
  return (
    <Interactive
      title="Shingles and MinHash"
      caption="Edit the two texts or pick an example. Each text becomes its set of k-shingles (runs of k words or k characters). The resemblance J is the Jaccard similarity of the two sets. MinHash keeps, for each hash function, the smallest hash in each set; the fraction of functions whose minima agree estimates J. The estimate wanders early and settles inside the ±2 standard-deviation band √(J(1 − J)/K) as functions are added."
      controls={
        <>
          <ParamChoice
            label="example"
            value={preset}
            onChange={choose}
            options={[
              ...Object.entries(PRESETS).map(([key, v]) => ({ value: key as Preset, label: v.label })),
              { value: 'custom', label: 'custom' },
            ]}
          />
          <ParamChoice
            label="shingle unit"
            value={unit}
            onChange={setUnit}
            options={[
              { value: 'word', label: 'words' },
              { value: 'char', label: 'characters' },
            ]}
          />
          <ParamSlider label="shingle size k" value={k} onChange={setK} min={1} max={10} step={1} debounceMs={0} />
          <ParamSlider
            label="hash functions K"
            value={K}
            onChange={setKHashes}
            min={1}
            max={MAX_HASHES}
            step={1}
            format={(v) => v.toFixed(0)}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>new hash functions</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="|S(A)|" value={m.sa.size} />
          <Readout label="|S(B)|" value={m.sb.size} />
          <Readout label="shared" value={m.inter} />
          <Readout label="resemblance J" value={formatNumber(m.J)} />
          <Readout label="containment of A in B" value={formatNumber(m.sa.size ? m.inter / m.sa.size : 0)} />
          <Readout label={`MinHash estimate, K = ${K}`} value={formatNumber(estimate)} />
          <Readout label="standard deviation" value={formatNumber(Math.sqrt((m.J * (1 - m.J)) / K))} />
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs text-muted-foreground">
          text A
          <Textarea
            value={a}
            onChange={(e) => {
              setA(e.target.value)
              setPreset('custom')
            }}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
        <label className="space-y-1 text-xs text-muted-foreground">
          text B
          <Textarea
            value={b}
            onChange={(e) => {
              setB(e.target.value)
              setPreset('custom')
            }}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
      </div>
      <XYChart
        series={series}
        xLabel="number of hash functions"
        yLabel="resemblance"
        xRange={[1, Math.max(2, K)]}
        yRange={[0, 1]}
        height={260}
      />
    </Interactive>
  )
}
