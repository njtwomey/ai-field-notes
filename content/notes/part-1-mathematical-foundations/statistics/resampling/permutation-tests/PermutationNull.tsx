import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { studentTCdf } from 'aifn-compute/numerics/special'

type Shape = 'normal' | 'skewed'

const BINS = 41

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const variance = (xs: number[]) => {
  const m = mean(xs)
  return xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1)
}

/**
 * Two samples of size n; group B is shifted by δ. The permutation null is the distribution of the difference in means
 * over random relabellings of the pooled data; the p-value counts relabellings at least as extreme as the observed one.
 */
export function PermutationNull() {
  const state = useFigureState({
    delta: float(1, { min: 0, max: 2, step: 0.1, label: 'shift δ' }),
    n: int(8, { min: 3, max: 40, step: 1, label: 'size of each group n' }),
    perms: int(2000, { min: 200, max: 10000, step: 200, label: 'permutations' }),
    shape: choice<Shape>(
      [
        { value: 'normal', label: 'normal' },
        { value: 'skewed', label: 'skewed (exponential)' },
      ],
      'skewed',
      { label: 'data' },
    ),
    seed: int(3, { ge: 0, label: 'seed' }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const draw = () => (state.shape === 'normal' ? normal(g) : -Math.log(Math.max(uniform(g), 1e-12)) - 1)
    const a = Array.from({ length: state.n }, draw)
    const b = Array.from({ length: state.n }, () => draw() + state.delta)
    const obs = mean(b) - mean(a)
    const pooled = [...a, ...b]
    const total = pooled.reduce((s, x) => s + x, 0)
    const diffs: number[] = []
    const idx = pooled.map((_, i) => i)
    for (let p = 0; p < state.perms; p++) {
      // Partial Fisher–Yates shuffle: the first n positions become group B.
      for (let i = 0; i < state.n; i++) {
        const j = i + Math.floor(uniform(g) * (2 * state.n - i))
        ;[idx[i], idx[j]] = [idx[j], idx[i]]
      }
      let sb = 0
      for (let i = 0; i < state.n; i++) sb += pooled[idx[i]]
      diffs.push(sb / state.n - (total - sb) / state.n)
    }
    const extreme = diffs.filter((d) => Math.abs(d) >= Math.abs(obs) - 1e-12).length
    const pPerm = (extreme + 1) / (state.perms + 1)
    const va = variance(a) / state.n
    const vb = variance(b) / state.n
    const t = obs / Math.sqrt(va + vb)
    const df = (va + vb) ** 2 / (va ** 2 / (state.n - 1) + vb ** 2 / (state.n - 1))
    const pWelch = 2 * (1 - studentTCdf(Math.abs(t), df))

    const span = Math.max(Math.abs(obs), ...diffs.map(Math.abs)) * 1.05
    const width = (2 * span) / BINS
    const mids = Array.from({ length: BINS }, (_, i) => -span + (i + 0.5) * width)
    const inner = new Array<number>(BINS).fill(0)
    const tail = new Array<number>(BINS).fill(0)
    for (const d of diffs) {
      const k = Math.min(BINS - 1, Math.max(0, Math.floor((d + span) / width)))
      if (Math.abs(d) >= Math.abs(obs) - 1e-12) tail[k]++
      else inner[k]++
    }
    const scale = 1 / (state.perms * width)
    const top = Math.max(...mids.map((_, i) => (inner[i] + tail[i]) * scale))
    const series = [
      { name: 'permutation null', x: mids, y: inner.map((c) => c * scale), slot: 0 },
      { name: 'at least as extreme', x: mids, y: tail.map((c) => c * scale), slot: 1 },
      { name: 'observed difference', x: [obs, obs], y: [0, top * 1.1], emphasis: true },
      { name: 'mirror of observed', x: [-obs, -obs], y: [0, top * 1.1], emphasis: true, dashed: true },
    ] as const
    return { obs, pPerm, pWelch, series, span, top }
  }, [state.n, state.delta, state.perms, state.shape, state.seed])

  const xAxis = useAxis({ label: 'difference in means (B − A)', range: [-r.span, r.span] })
  const yAxis = useAxis({ label: 'density', range: [0, r.top * 1.15] })
  return (
    <Figure
      title="The permutation null distribution"
      state={state}
      caption="Two groups of size n, the second shifted by δ. Each permutation reassigns the pooled observations to the two groups at random and recomputes the difference in means. Under the null hypothesis every assignment is equally likely, so the histogram is the exact reference distribution up to Monte Carlo error. The p-value is the fraction of permutations at least as extreme as the observed difference (in either direction), counting the observed labelling itself. With skewed data and small n it can differ from Welch's t-test, which relies on approximate normality."

      readouts={
        <>
          <Readout label="observed difference" value={formatNumber(r.obs)} />
          <Readout label="permutation p" value={formatNumber(r.pPerm)} />
          <Readout label="Welch t-test p" value={formatNumber(r.pWelch)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Bars {...r.series[0]} />
        <Bars {...r.series[1]} />
        <Curve {...r.series[2]} />
        <Curve {...r.series[3]} />
      </Plot>
    </Figure>
  )
}
