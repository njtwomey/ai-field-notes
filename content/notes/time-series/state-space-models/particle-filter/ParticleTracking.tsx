import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

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

function simulate(seed: number) {
  const g = rng(seed)
  let z = Math.sqrt(P0) * g.normal()
  const truth: number[] = []
  const obs: number[] = []
  for (let t = 1; t <= STEPS; t++) {
    z = f(z, t) + Math.sqrt(Q) * g.normal()
    truth.push(z)
    obs.push(h(z) + Math.sqrt(R) * g.normal())
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
  const g = rng(seed)
  let particles = Array.from({ length: n }, () => Math.sqrt(P0) * g.normal())
  // Log-weights, so that without resampling they can fall far below the smallest double without underflowing.
  let logW = new Array<number>(n).fill(0)
  const mean: number[] = []
  const ess: number[] = []
  const density: number[][] = BIN_CENTRES.map(() => new Array<number>(STEPS).fill(0))
  for (let t = 1; t <= STEPS; t++) {
    particles = particles.map((z) => f(z, t) + Math.sqrt(Q) * g.normal())
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
    const idx = resample(cumulative, scheme, g.uniform)
    particles = idx.map((i) => particles[i])
    logW = new Array<number>(n).fill(0)
  }
  // Scale each time step's column to a maximum of 1, so that sharp and diffuse posteriors are both visible.
  for (let t = 0; t < STEPS; t++) {
    const peak = Math.max(...density.map((row) => row[t])) || 1
    density.forEach((row) => (row[t] /= peak))
  }
  return { mean, ess, density }
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
  const [n, setN] = useState('1000')
  const [scheme, setScheme] = useState<Scheme>('systematic')
  const [when, setWhen] = useState<When>('always')
  const seed = useParam(7, { min: 1, max: 20, step: 1 })

  const world = useMemo(() => simulate(seed.value), [seed.value])
  const ekf = useMemo(() => extendedKalman(world.obs), [world])
  const pf = useMemo(
    () => particleFilter(world.obs, Number(n), scheme, when, seed.value + 1000),
    [world, n, scheme, when, seed.value],
  )

  const t = world.truth.map((_, i) => i + 1)
  // The EKF can leave the plotted range; clip it to the frame.
  const clip = (v: number) => Math.max(-30, Math.min(30, v))
  const overlay: HeatmapOverlay[] = [
    { name: 'true state', type: 'line', x: t, y: world.truth, slot: 1 },
    { name: 'particle filter mean', type: 'line', x: t, y: pf.mean, slot: 2 },
    { name: 'EKF mean', type: 'line', x: t, y: ekf.map(clip), slot: 3 },
  ]
  const essSeries: XYSeries[] = [{ name: 'effective sample size / N', type: 'line', x: t, y: pf.ess, slot: 0 }]

  return (
    <Interactive
      title="Particles tracking a nonlinear state"
      caption="The state follows strongly nonlinear dynamics and is observed through z²/20 plus noise, so each observation fits both z and −z. The shading is the particle filter's estimate of the filtering distribution at each step (each column scaled to a maximum of 1). It is often bimodal, with modes at ±z, and the filter keeps both until the dynamics break the tie. The extended Kalman filter keeps one Gaussian and frequently locks onto the wrong sign. Without resampling, the weights collapse onto a few particles within a few steps."
      controls={
        <>
          <ParamChoice
            label="particles N"
            value={n}
            onChange={setN}
            options={[
              { value: '10', label: '10' },
              { value: '100', label: '100' },
              { value: '1000', label: '1000' },
              { value: '5000', label: '5000' },
            ]}
          />
          <ParamChoice
            label="resample"
            value={when}
            onChange={setWhen}
            options={[
              { value: 'always', label: 'every step' },
              { value: 'adaptive', label: 'ESS < N/2' },
              { value: 'never', label: 'never' },
            ]}
          />
          <ParamChoice
            label="resampling scheme"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'systematic', label: 'systematic' },
              { value: 'multinomial', label: 'multinomial' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="RMSE, particle filter" value={formatNumber(rmse(pf.mean, world.truth))} />
          <Readout label="RMSE, EKF" value={formatNumber(rmse(ekf, world.truth))} />
          <Readout label="mean ESS / N" value={formatNumber(pf.ess.reduce((a, b) => a + b, 0) / pf.ess.length)} />
        </>
      }
    >
      <div className="space-y-4">
        <Heatmap
          x={t}
          y={BIN_CENTRES}
          z={pf.density}
          range={[0, 1]}
          overlay={overlay}
          xLabel="step t"
          yLabel="state z"
          valueLabel="filtering density (column max 1)"
          height={320}
        />
        <XYChart series={essSeries} xLabel="step t" yLabel="ESS / N" yRange={[0, 1]} height={160} />
      </div>
    </Interactive>
  )
}
