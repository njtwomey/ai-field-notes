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
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 480

type Scenario = 'motif' | 'discord'

/**
 * Two synthetic series. `motif`: mild noise with one distinctive pattern planted twice, so the matrix profile's minimum
 * finds the pair. `discord`: a repeating heartbeat-like signal with one abnormal beat, so every normal beat has a close
 * neighbour and the abnormal one stands out as the maximum. Each scenario isolates one idea: in plain noise every
 * stretch is unusual, and in a repeating signal every stretch is a motif.
 */
function series(scenario: Scenario, seed: number) {
  const g = stream(seed)
  if (scenario === 'motif') {
    const x: number[] = []
    let level = 0
    for (let t = 0; t < N; t++) {
      level += 0.02 * normal(g)
      x.push(level + 0.4 * normal(g))
    }
    const shape = Array.from({ length: 40 }, (_, k) => {
      const u = k / 40
      return 2.5 * Math.exp(-(((u - 0.3) / 0.1) ** 2)) - 1.8 * Math.exp(-(((u - 0.7) / 0.06) ** 2))
    })
    const planted = [70, 300]
    for (const a of planted) shape.forEach((v, k) => (x[a + k] += v))
    return { x, planted }
  }
  const beat = (length: number, abnormal: boolean) =>
    Array.from({ length }, (_, k) => {
      const u = k / (length - 1)
      const p = 0.15 * Math.exp(-(((u - 0.2) / 0.05) ** 2))
      return abnormal
        ? p + 0.6 * Math.exp(-(((u - 0.4) / 0.06) ** 2)) - 0.3 * Math.exp(-(((u - 0.55) / 0.06) ** 2))
        : p +
            Math.exp(-(((u - 0.35) / 0.015) ** 2)) -
            0.25 * Math.exp(-(((u - 0.39) / 0.02) ** 2)) +
            0.3 * Math.exp(-(((u - 0.65) / 0.07) ** 2))
    })
  const x: number[] = []
  let abnormalAt = 0
  for (let i = 0; x.length < N; i++) {
    const length = 40 + Math.floor(uniform(g) * 7) - 3
    if (i === 6) abnormalAt = x.length
    x.push(...beat(length, i === 6))
  }
  return { x: x.slice(0, N).map((v) => v + 0.03 * normal(g)), planted: [abnormalAt] }
}

/** Naive matrix profile, O(n² m): fine at this size, and the definition is visible in the code. */
function matrixProfile(x: number[], m: number) {
  const n = x.length - m + 1
  const mean: number[] = []
  const sd: number[] = []
  for (let i = 0; i < n; i++) {
    let s = 0
    let s2 = 0
    for (let k = 0; k < m; k++) {
      s += x[i + k]
      s2 += x[i + k] ** 2
    }
    const mu = s / m
    mean.push(mu)
    sd.push(Math.sqrt(Math.max(s2 / m - mu * mu, 1e-12)))
  }
  // Exclusion zone: neighbours closer than m/4 overlap too much to count as a match (trivial matches).
  const zone = Math.ceil(m / 4)
  const profile = new Array<number>(n).fill(Infinity)
  const index = new Array<number>(n).fill(-1)
  for (let i = 0; i < n; i++) {
    for (let j = i + zone + 1; j < n; j++) {
      let dot = 0
      for (let k = 0; k < m; k++) dot += x[i + k] * x[j + k]
      // d² = 2m (1 − ρ) with ρ the Pearson correlation of the two subsequences.
      const rho = (dot - m * mean[i] * mean[j]) / (m * sd[i] * sd[j])
      const d = Math.sqrt(Math.max(2 * m * (1 - rho), 0))
      if (d < profile[i]) {
        profile[i] = d
        index[i] = j
      }
      if (d < profile[j]) {
        profile[j] = d
        index[j] = i
      }
    }
  }
  return { profile, index }
}

export function MatrixProfileExplorer() {
  const state = useFigureState({
    scenario: choice<Scenario>(
      [
        { value: 'motif', label: 'pattern planted twice' },
        { value: 'discord', label: 'heartbeat with one abnormal beat' },
      ],
      'motif',
      { label: 'series' },
    ),
    m: int(40, { min: 8, max: 80, step: 4, label: 'subsequence length m', format: (v) => String(v) }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const s = series(state.scenario, state.seed)
    const { profile, index } = matrixProfile(s.x, state.m)
    let motif = 0
    let discord = 0
    profile.forEach((v, i) => {
      if (v < profile[motif]) motif = i
      if (v > profile[discord]) discord = i
    })
    return { ...s, profile, index, motif, pair: index[motif], discord }
  }, [state.scenario, state.m, state.seed])

  const t = r.x.map((_, i) => i)
  const span = (start: number) => Array.from({ length: state.m }, (_, k) => start + k)
  const highlight = (name: string, start: number, slot: number): SeriesSpec => ({
    name,
    type: 'line',
    x: span(start),
    y: span(start).map((i) => r.x[i]),
    slot,
  })
  const top: SeriesSpec[] = [
    { name: 'series', type: 'line', x: t, y: r.x, muted: true },
    highlight('motif pair', r.motif, 1),
    highlight('motif pair', r.pair, 1),
    highlight('discord', r.discord, 2),
  ]
  const bottom = [
    { name: 'matrix profile', x: r.profile.map((_, i) => i), y: r.profile, slot: 0 },
    {
      name: 'motif (minimum)',
      x: [r.motif, r.pair],
      y: [r.profile[r.motif], r.profile[r.pair]],
      slot: 1,
    },
    { name: 'discord (maximum)', x: [r.discord], y: [r.profile[r.discord]], slot: 2 },
  ] as const

  const xAxis = useAxis({ label: 'time', hold: 'union' })
  const yAxis = useAxis({ label: 'x', hold: 'union' })
  const xAxis2 = useAxis({ label: 'subsequence start i', hold: 'union' })
  const yAxis2 = useAxis({ label: 'P[i]', hold: 'union' })
  return (
    <Figure
      title="Motifs and discords from one profile"
      state={state}
      caption="Top: a time series. Bottom: its matrix profile, the distance from each length-m subsequence to its nearest non-overlapping neighbour after z-normalisation. With a pattern planted twice in noise, the two lowest points of the profile find the pair: the top motif. With a repeating heartbeat-like signal, every normal beat has a close neighbour, so the one abnormal beat is the highest point: the top discord. The window may land a few samples off the planted start when the pattern has quiet edges. Change m: too short and chance fragments match, far too long and the pattern is diluted."

      readouts={
        state.scenario === 'motif' ? (
          <>
            <Readout label="motif pair found at" value={`${Math.min(r.motif, r.pair)}, ${Math.max(r.motif, r.pair)}`} />
            <Readout label="pattern planted at" value={r.planted.join(', ')} />
            <Readout label="motif distance" value={formatNumber(r.profile[r.motif])} />
          </>
        ) : (
          <>
            <Readout label="discord found at" value={r.discord} />
            <Readout label="abnormal beat starts at" value={r.planted[0]} />
            <Readout label="discord distance" value={formatNumber(r.profile[r.discord])} />
          </>
        )
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={220}>
          {seriesLayers(top)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          <Curve {...bottom[0]} />
          <Points {...bottom[1]} />
          <Points {...bottom[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
