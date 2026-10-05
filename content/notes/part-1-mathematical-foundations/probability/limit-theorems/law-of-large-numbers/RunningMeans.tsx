import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
import type { Stream } from 'aifn-compute/foundation/random'

type DistId = 'bernoulli' | 'exponential' | 'cauchy'

const DISTRIBUTIONS: { value: DistId; label: string }[] = [
  { value: 'bernoulli', label: 'Bernoulli(0.3)' },
  { value: 'exponential', label: 'exponential(1)' },
  { value: 'cauchy', label: 'Cauchy' },
]

/** Mean and standard deviation of each distribution; the Cauchy has neither. */
const MOMENTS: Record<DistId, { mu: number; sigma: number } | null> = {
  bernoulli: { mu: 0.3, sigma: Math.sqrt(0.3 * 0.7) },
  exponential: { mu: 1, sigma: 1 },
  cauchy: null,
}

const MAX_N = 4000
/** Plot every STRIDE-th running mean, so each path has at most a few hundred points. */
const STRIDE = 10

function draw(id: DistId, u: () => number): number {
  if (id === 'bernoulli') return u() < 0.3 ? 1 : 0
  if (id === 'exponential') return -Math.log(1 - u())
  // Standard Cauchy by inverse transform; its median is 0 (it has no mean).
  return Math.tan(Math.PI * (u() - 0.5))
}

/** Running means of independent draws: they settle on the mean when it exists and never settle for the Cauchy. */
export function RunningMeans() {
  const state = useFigureState({
    id: choice<DistId>(DISTRIBUTIONS, 'exponential', { label: 'distribution' }),
    count: int(12, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
    n: int(1000, { min: 50, max: MAX_N, step: 50, label: 'n (draws shown)' }),
  })

  // All paths to MAX_N are simulated once per distribution and count; the n slider only changes how much is shown.
  // Path p has its own stream, so adding paths leaves the existing ones unchanged.
  const paths = useMemo(() => {
    const out: { x: number[]; y: number[] }[] = []
    for (let p = 0; p < state.count; p++) {
      const g: Stream = stream(1000 + p)
      const uniform = () => drawUniform(g)
      let s = 0
      const x: number[] = []
      const y: number[] = []
      for (let i = 1; i <= MAX_N; i++) {
        s += draw(state.id, uniform)
        if (i === 1 || i % STRIDE === 0) {
          x.push(i)
          y.push(s / i)
        }
      }
      out.push({ x, y })
    }
    return out
  }, [state.id, state.count])

  const m = MOMENTS[state.id]
  const center = m ? m.mu : 0
  const half = m ? 4 * m.sigma : 6
  const series = useMemo((): SeriesSpec[] => {
    const many = paths.length > 1
    const shown = paths.map((p): SeriesSpec => {
      const cut = p.x.findIndex((v) => v > state.n)
      const end = cut === -1 ? p.x.length : cut
      const name = many ? 'running means' : 'running mean'
      return { name, type: 'line', x: p.x.slice(0, end), y: p.y.slice(0, end), slot: 1, thin: many }
    })
    if (!m) return shown
    const grid = Array.from({ length: 200 }, (_, i) => 1 + ((state.n - 1) * i) / 199)
    return [
      ...shown,
      { name: 'μ + 2σ/√n', type: 'line', x: grid, y: grid.map((k) => m.mu + (2 * m.sigma) / Math.sqrt(k)), slot: 0 },
      { name: 'μ − 2σ/√n', type: 'line', x: grid, y: grid.map((k) => m.mu - (2 * m.sigma) / Math.sqrt(k)), slot: 0 },
      { name: 'μ', type: 'line', x: [1, state.n], y: [m.mu, m.mu], emphasis: true, dashed: true },
    ]
  }, [paths, state.n, m])

  const finals = paths.map((p) => {
    const cut = p.x.findIndex((v) => v > state.n)
    return p.y[(cut === -1 ? p.x.length : cut) - 1]
  })
  const spread = Math.max(...finals) - Math.min(...finals)

  const xAxis = useAxis({ label: 'n', range: [1, state.n] })
  const yAxis = useAxis({ label: 'running mean', range: [center - half, center + half] })
  return (
    <Figure
      title="Running means of independent draws"
      state={state}
      caption="Independent sequences, each drawn as a light line showing the mean of its first n draws; the paths slider sets how many sequences. For the Bernoulli and exponential distributions the paths close in on μ inside the band μ ± 2σ/√n. The Cauchy distribution has no mean, and its running means keep jumping however large n is."

      readouts={
        <>
          <Readout label="mean μ" value={m ? formatNumber(m.mu) : 'undefined'} />
          <Readout label={`spread of the ${state.count} means at n`} value={formatNumber(spread)} />
          {m && <Readout label="4σ/√n" value={formatNumber((4 * m.sigma) / Math.sqrt(state.n))} />}
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
