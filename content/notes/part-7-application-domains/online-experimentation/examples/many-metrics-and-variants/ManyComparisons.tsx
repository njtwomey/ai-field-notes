import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'

type Correction = 'none' | 'bonferroni' | 'holm' | 'bh'

const EXPERIMENTS = 2000
const MAX_COMPARISONS = 100
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]
const CORRECTIONS: { value: Correction; label: string }[] = [
  { value: 'none', label: 'none' },
  { value: 'bonferroni', label: 'Bonferroni' },
  { value: 'holm', label: 'Holm' },
  { value: 'bh', label: 'Benjamini–Hochberg' },
]

/** Which of the p-values a procedure rejects at level α. */
function reject(p: number[], alpha: number, method: Correction): boolean[] {
  const n = p.length
  if (method === 'none') return p.map((q) => q < alpha)
  if (method === 'bonferroni') return p.map((q) => q < alpha / n)
  const order = p.map((q, i) => [q, i] as const).sort((a, b) => a[0] - b[0])
  const out = new Array<boolean>(n).fill(false)
  if (method === 'holm') {
    // Step down: reject the i-th smallest while it is below α/(n − i + 1); stop at the first that is not.
    for (let i = 0; i < n && order[i][0] < alpha / (n - i); i++) out[order[i][1]] = true
    return out
  }
  // Benjamini–Hochberg: reject every p-value up to the largest rank i with p₍ᵢ₎ ≤ iα/n.
  let last = -1
  order.forEach(([q], i) => {
    if (q <= ((i + 1) * alpha) / n) last = i
  })
  for (let i = 0; i <= last; i++) out[order[i][1]] = true
  return out
}

/**
 * Null A/B/n experiments: k variants, each compared with one shared control on m metrics, with no real effects. The
 * chance that at least one comparison looks significant grows quickly with the number of comparisons.
 */
export function ManyComparisons() {
  const state = useFigureState({
    metrics: int(10, { min: 1, max: 20, step: 1, label: 'metrics m' }),
    variants: int(3, { min: 1, max: 5, step: 1, label: 'variants k' }),
    alpha: choice(ALPHAS, '0.05', { label: 'α' }),
    method: choice<Correction>(CORRECTIONS, 'none', { label: 'correction' }),
    shown: int(1, { min: 1, max: 40, step: 1, label: 'experiment shown' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const a = Number(state.alpha)
  const m = state.metrics
  const k = state.variants
  const n = m * k

  const sim = useMemo(() => {
    const g = stream(state.seed)
    // z for variant v on metric j is (X_vj − C_j)/√2: variants share the control's noise C_j, as in a real A/B/n test.
    return Array.from({ length: EXPERIMENTS }, () => {
      const control = Array.from({ length: m }, () => normal(g))
      return Array.from({ length: k }, () => control.map((c) => (normal(g) - c) / Math.SQRT2)).flat()
    })
  }, [m, k, state.seed])

  const r = useMemo(() => {
    const pValues = sim.map((zs) => zs.map((z) => 2 * (1 - normalCdf(Math.abs(z)))))
    const anyRaw = pValues.filter((p) => p.some((q) => q < a)).length / EXPERIMENTS
    const anyCorrected = pValues.filter((p) => reject(p, a, state.method).some(Boolean)).length / EXPERIMENTS
    const example = pValues[state.shown - 1]
    return { anyRaw, anyCorrected, example, flags: reject(example, a, state.method) }
  }, [sim, a, state.method, state.shown])

  // Comparisons are laid out by metric, with the variants side by side within each metric.
  const position = (i: number) => {
    const variant = Math.floor(i / m)
    const metric = i % m
    return metric + 1 + (variant - (k - 1) / 2) * (0.6 / Math.max(k, 1))
  }
  const logp = r.example.map((p) => -Math.log10(p))
  const idx = r.example.map((_, i) => i)
  const hits = idx.filter((i) => r.flags[i])
  const misses = idx.filter((i) => !r.flags[i])
  const detail = [
    { name: 'not significant', x: misses.map(position), y: misses.map((i) => logp[i]), muted: true },
    { name: 'false positive', x: hits.map(position), y: hits.map((i) => logp[i]), slot: 1 },
    {
      name: 'α, uncorrected',
      x: [0.5, m + 0.5],
      y: [-Math.log10(a), -Math.log10(a)],
      slot: 0,
      dashed: true,
    },
    {
      name: 'α / (mk), Bonferroni',
      x: [0.5, m + 0.5],
      y: [-Math.log10(a / n), -Math.log10(a / n)],
      slot: 2,
      dashed: true,
    },
  ] as const

  const counts = toFlat(linspace(1, MAX_COMPARISONS, MAX_COMPARISONS))
  const curve: SeriesSpec[] = [
    {
      name: '1 − (1 − α)ⁿ, independent tests',
      type: 'line',
      x: counts,
      y: counts.map((c) => 1 - (1 - a) ** c),
      slot: 0,
    },
    { name: 'α', type: 'line', x: [1, MAX_COMPARISONS], y: [a, a], slot: 2, dashed: true },
    { name: 'simulated, uncorrected', type: 'scatter', x: [n], y: [r.anyRaw], slot: 1 },
    ...(state.method === 'none'
      ? []
      : [{ name: 'simulated, corrected', type: 'scatter' as const, x: [n], y: [r.anyCorrected], emphasis: true }]),
  ]

  const xAxis = useAxis({ label: 'metric', hold: 'union' })
  const yAxis = useAxis({ label: '−log₁₀ p', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'comparisons mk', hold: 'union' })
  const yAxis2 = useAxis({ label: 'P(at least one false positive)', range: [0, 1] })
  return (
    <Figure
      title="Search enough comparisons and one will win"
      state={state}
      caption="Every comparison here is null: no variant changes any metric. The left chart shows one experiment's p-values on a −log₁₀ scale, one column per metric with the variants side by side; points above the line are declared significant. The right chart shows how often an experiment produces at least one false positive, against the number of comparisons mk. Variants share a control, so their comparisons are correlated and the rate sits a little below the independence curve. Choose a correction to bring the rate back to α."

      readouts={
        <>
          <Readout label="comparisons mk" value={n} />
          <Readout label="1 − (1 − α)^mk" value={formatNumber(1 - (1 - a) ** n)} />
          <Readout label="P(≥ 1 false positive), uncorrected" value={formatNumber(r.anyRaw)} />
          <Readout
            label={`P(≥ 1 false positive), ${state.method === 'none' ? 'no correction' : CORRECTIONS.find((c) => c.value === state.method)!.label}`}
            value={formatNumber(r.anyCorrected)}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Points {...detail[0]} />
          <Points {...detail[1]} />
          <Curve {...detail[2]} />
          <Curve {...detail[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          {seriesLayers(curve)}
        </Plot>
      </div>
    </Figure>
  )
}
