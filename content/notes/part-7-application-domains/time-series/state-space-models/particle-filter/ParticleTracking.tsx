import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal as drawNormal, stream, uniform as drawUniform } from 'aifn/foundation/random'

const STEPS = 50
const Q = 10
const R = 1
const P0 = 5

/** Univariate nonstationary growth model: strongly nonlinear dynamics, and an observation that cannot see the sign. */
const f = (z: number, t: number) => z / 2 + (25 * z) / (1 + z * z) + 8 * Math.cos(1.2 * t)
const h = (z: number) => (z * z) / 20

type Scheme = 'multinomial' | 'systematic'
type When = 'always' | 'adaptive' | 'never'

const BIN_CENTRES = Array.from({ length: 61 }, (_, i) => i - 30)
const T = Array.from({ length: STEPS }, (_, i) => i + 1)
/** Lines that leave the plotted range (the EKF, stray lineages) are clipped to the frame. */
const clip = (v: number) => Math.max(-30, Math.min(30, v))

function simulate(seed: number) {
  const g = stream(seed)
  let z = Math.sqrt(P0) * drawNormal(g)
  const truth: number[] = []
  const obs: number[] = []
  for (let t = 1; t <= STEPS; t++) {
    z = f(z, t) + Math.sqrt(Q) * drawNormal(g)
    truth.push(z)
    obs.push(h(z) + Math.sqrt(R) * drawNormal(g))
  }
  return { truth, obs }
}

/** Indices drawn in proportion to the weights. `cumulative` is the running sum of the normalised weights. */
function resample(cumulative: number[], scheme: Scheme, uniform: () => number): number[] {
  const n = cumulative.length
  // Multinomial: n independent uniforms, sorted. Systematic: one uniform, then evenly spaced.
  const u =
    scheme === 'multinomial'
      ? Array.from({ length: n }, uniform).sort((a, b) => a - b)
      : Array.from({ length: n }, (_, i) => (uniform() + i) / n)
  const out: number[] = []
  let j = 0
  for (const v of u) {
    while (j < n - 1 && cumulative[j] < v) j++
    out.push(j)
  }
  return out
}

/** Bootstrap particle filter: propagate through the dynamics, weight by the likelihood, resample. */
function particleFilter(obs: number[], n: number, scheme: Scheme, when: When, seed: number) {
  const g = stream(seed)
  let particles = Array.from({ length: n }, () => Math.sqrt(P0) * drawNormal(g))
  // Log-weights, so that without resampling they can fall far below the smallest double without underflowing.
  let logW = new Array<number>(n).fill(0)
  // Ancestry, for drawing lineages: history[t][i] is particle i at step t + 1 after propagation, and parent[t][i] is
  // the index at step t of the particle it descends from (identity when the previous step did not resample).
  const history: number[][] = []
  const parent: number[][] = []
  let lastIdx: number[] = Array.from({ length: n }, (_, i) => i)
  const mean: number[] = []
  const ess: number[] = []
  const density: number[][] = BIN_CENTRES.map(() => new Array<number>(STEPS).fill(0))
  for (let t = 1; t <= STEPS; t++) {
    particles = particles.map((z) => f(z, t) + Math.sqrt(Q) * drawNormal(g))
    history.push(particles)
    parent.push(lastIdx)
    lastIdx = Array.from({ length: n }, (_, i) => i)
    logW = logW.map((lw, i) => lw - (obs[t - 1] - h(particles[i])) ** 2 / (2 * R))
    const top = Math.max(...logW)
    const w = logW.map((lw) => Math.exp(lw - top))
    const total = w.reduce((a, b) => a + b, 0)
    for (let i = 0; i < n; i++) w[i] /= total
    mean.push(particles.reduce((a, z, i) => a + w[i] * z, 0))
    const effective = 1 / w.reduce((a, v) => a + v * v, 0)
    ess.push(effective / n)
    // Weighted histogram of the filtering distribution, one bin per unit of z.
    particles.forEach((z, i) => {
      const b = Math.round(z) + 30
      if (b >= 0 && b < BIN_CENTRES.length) density[b][t - 1] += w[i]
    })
    if (when === 'never' || (when === 'adaptive' && effective >= n / 2)) {
      logW = w.map(Math.log)
      continue
    }
    const cumulative: number[] = []
    w.reduce((a, v, i) => (cumulative[i] = a + v), 0)
    const idx = resample(cumulative, scheme, () => drawUniform(g))
    lastIdx = idx
    particles = idx.map((i) => particles[i])
    logW = new Array<number>(n).fill(0)
  }
  // Scale each time step's column to a maximum of 1, so that sharp and diffuse posteriors are both visible.
  for (let t = 0; t < STEPS; t++) {
    const peak = Math.max(...density.map((row) => row[t])) || 1
    density.forEach((row) => (row[t] /= peak))
  }
  return { mean, ess, density, history, parent }
}

/**
 * Ancestral lineages of `count` particles spread evenly through the final population, traced back to step 1 through
 * the resampling indices. Repeated resampling makes them coalesce into a few ancestors (path degeneracy).
 */
function lineages(history: number[][], parent: number[][], count: number): number[][] {
  const n = history[0].length
  const k = Math.min(count, n)
  return Array.from({ length: k }, (_, j) => {
    let i = Math.floor((j * n) / k)
    const path = new Array<number>(history.length)
    for (let t = history.length - 1; t >= 0; t--) {
      path[t] = history[t][i]
      // parent[t] maps particles at step t + 1 to their parents at step t (the initial draws when t = 0).
      i = parent[t][i]
    }
    return path
  })
}

/** Extended Kalman filter on the same model, linearised at the current mean. */
function extendedKalman(obs: number[]) {
  let m = 0
  let P = P0
  return obs.map((x, k) => {
    const t = k + 1
    const F = 0.5 + (25 * (1 - m * m)) / (1 + m * m) ** 2
    m = f(m, t)
    P = F * F * P + Q
    const H = m / 10
    const K = (P * H) / (H * H * P + R)
    m += K * (x - h(m))
    P *= 1 - K * H
    return m
  })
}

const rmse = (est: number[], truth: number[]) =>
  Math.sqrt(est.reduce((a, e, t) => a + (e - truth[t]) ** 2, 0) / est.length)

export function ParticleTracking() {
  const state = useFigureState({
    n: choice(
      [
        { value: '10', label: '10' },
        { value: '100', label: '100' },
        { value: '1000', label: '1000' },
        { value: '5000', label: '5000' },
      ],
      '1000',
      { label: 'particles N' },
    ),
    when: choice<When>(
      [
        { value: 'always', label: 'every step' },
        { value: 'adaptive', label: 'ESS < N/2' },
        { value: 'never', label: 'never' },
      ],
      'always',
      { label: 'resample' },
    ),
    scheme: choice<Scheme>(
      [
        { value: 'systematic', label: 'systematic' },
        { value: 'multinomial', label: 'multinomial' },
      ],
      'systematic',
      { label: 'resampling scheme' },
    ),
    shown: int(10, { min: 1, max: 50, step: 1, label: 'lineages', format: (v) => String(v) }),
    seed: int(7, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const world = useMemo(() => simulate(state.seed), [state.seed])
  const ekf = useMemo(() => extendedKalman(world.obs), [world])
  const pf = useMemo(
    () => particleFilter(world.obs, Number(state.n), state.scheme, state.when, state.seed + 1000),
    [world, state.n, state.scheme, state.when, state.seed],
  )

  const lines = useMemo(() => lineages(pf.history, pf.parent, state.shown), [pf, state.shown])

  const overlay: SeriesSpec[] = useMemo(
    () => [
      ...lines.map((y): SeriesSpec => ({
        name: 'particle lineages',
        type: 'line',
        x: T,
        y: y.map(clip),
        slot: 4,
        thin: lines.length > 1,
      })),
      { name: 'true state', type: 'line', x: T, y: world.truth, slot: 1 },
      { name: 'particle filter mean', type: 'line', x: T, y: pf.mean, slot: 2 },
      { name: 'EKF mean', type: 'line', x: T, y: ekf.map(clip), slot: 3 },
    ],
    [lines, world, pf, ekf],
  )
  const essSeries = [{ name: 'effective sample size / N', x: T, y: pf.ess, slot: 0 }] as const

  const xAxis = useAxis({ label: 'step t' })
  const yAxis = useAxis({ label: 'state z' })
  const xAxis2 = useAxis({ label: 'step t', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ESS / N', range: [0, 1] })
  return (
    <Figure
      title="Particles tracking a nonlinear state"
      state={state}
      caption="The state follows strongly nonlinear dynamics and is observed through z²/20 plus noise, so each observation fits both z and −z. The shading is the particle filter's estimate of the filtering distribution at each step (each column scaled to a maximum of 1). It is often bimodal, with modes at ±z, and the filter keeps both until the dynamics break the tie. The extended Kalman filter keeps one Gaussian and frequently locks onto the wrong sign. Without resampling, the weights collapse onto a few particles within a few steps. The light lines are the ancestral lineages of particles spread evenly through the final population, traced back through the resampling steps; the lineages slider sets how many are drawn (at most N). With resampling at every step they coalesce into one or a few ancestors a few steps back: this is path degeneracy."

      readouts={
        <>
          <Readout label="RMSE, particle filter" value={formatNumber(rmse(pf.mean, world.truth))} />
          <Readout label="RMSE, EKF" value={formatNumber(rmse(ekf, world.truth))} />
          <Readout label="mean ESS / N" value={formatNumber(pf.ess.reduce((a, b) => a + b, 0) / pf.ess.length)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Raster x={T} y={BIN_CENTRES} z={pf.density} range={[0, 1]} valueLabel={'filtering density (column max 1)'} />
          {seriesLayers(overlay, { live: true })}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={160}>
          <Curve {...essSeries[0]} />
        </Plot>
      </div>
    </Figure>
  )
}
