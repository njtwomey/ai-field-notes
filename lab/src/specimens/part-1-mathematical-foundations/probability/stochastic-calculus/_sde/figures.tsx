import { densityEvolution, gridPoints } from 'aifn-methods/dynamics/pde'
import { stream } from 'aifn-compute/foundation/random'
import {
  eulerMaruyama,
  geometricBrownianMotion,
  milstein,
  ornsteinUhlenbeck,
  paths,
  stochasticRungeKutta,
  type Sde,
  type SdeOptions,
  type SdeState,
} from 'aifn-compute/dynamics/sde'
import { histogram } from 'aifn-compute/probability/stats'
import { mul, sub, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { run, trace, type Algorithm } from 'aifn-compute/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useComputed, useFigureState } from 'aifn-render/state'
import { Curve, Histogram, Plot, Plots, Points, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => formatNumber(v)

type Process = {
  value: string
  label: string
  sde: Sde
  x0: number
  tEnd: number
  /** The drift and noise as scalar functions, for the Fokker–Planck density. */
  drift: (x: number) => number
  sigma: (x: number) => number
  range: [number, number]
}

const OU = ornsteinUhlenbeck({ theta: 1.5, mu: 0, sigma: 0.8 })
const GBM = geometricBrownianMotion({ mu: 0.3, sigma: 0.5 })

const PROCESSES: Process[] = [
  {
    value: 'double-well',
    label: 'double well dX = (X − X³) dt + 0.7 dW',
    sde: { drift: (_t, x) => sub(x, mul(x, mul(x, x))), diffusion: () => 0.7 },
    x0: 0.1,
    tEnd: 6,
    drift: (x) => x - x ** 3,
    sigma: () => 0.7,
    range: [-2.2, 2.2],
  },
  {
    value: 'ou',
    label: 'Ornstein–Uhlenbeck dX = −1.5X dt + 0.8 dW',
    sde: OU.sde,
    x0: 2,
    tEnd: 3,
    drift: (x) => -1.5 * x,
    sigma: () => 0.8,
    range: [-2, 2.8],
  },
  {
    value: 'gbm',
    label: 'geometric Brownian motion dX = 0.3X dt + 0.5X dW',
    sde: GBM.sde,
    x0: 1,
    tEnd: 2,
    drift: (x) => 0.3 * x,
    sigma: (x) => 0.5 * Math.abs(x),
    range: [0, 6],
  },
]

type SchemeName = 'euler-maruyama' | 'milstein' | 'stochastic-runge-kutta'
const SCHEMES: {
  value: SchemeName
  label: string
  make: (s: Sde, o: SdeOptions) => Algorithm<{ x0: number; paths: number }, SdeState>
}[] = [
  { value: 'euler-maruyama', label: 'Euler–Maruyama', make: eulerMaruyama },
  { value: 'milstein', label: 'Milstein', make: milstein },
  { value: 'stochastic-runge-kutta', label: 'stochastic Runge–Kutta', make: stochasticRungeKutta },
]
const COUNTS = [10, 50, 200, 1000, 3000].map((n) => ({ value: n, label: `${n} paths` }))

const BINS = 40
const DRAWN = 300

export function PathCloudSpecimen() {
  const state = useFigureState({
    process: row('1 · process', {
      which: choice(
        PROCESSES.map(({ value, label }) => ({ value, label })),
        'double-well',
        { label: 'SDE' },
      ),
    }),
    simulation: row('2 · simulation', {
      scheme: choice(
        SCHEMES.map(({ value, label }) => ({ value, label })),
        'euler-maruyama',
        { label: 'scheme' },
      ),
      n: choice(COUNTS, 200, { label: 'paths' }),
      h: slider(0.002, 0.2, 0.02, { label: 'step h', step: 0.002 }),
    }),
    reveal: row('3 · reveal', { showDensity: toggle(true, 'Fokker–Planck density') }),
  })
  const which = state.process.which
  const scheme: SchemeName = state.simulation.scheme
  const { n, h } = state.simulation
  const showDensity = state.reveal.showDensity
  const proc = PROCESSES.find((p) => p.value === which)!
  const make = SCHEMES.find((s) => s.value === scheme)!.make
  // Thousands of paths at small h take longer than a frame: a slider drag reruns on release.
  const sim = useComputed(
    () => {
      const tr = trace(make(proc.sde, { stepSize: h, tEnd: proc.tEnd }), { x0: proc.x0, paths: n }, 100_000, {
        stream: stream('path-cloud'),
        every: Math.max(1, Math.round(proc.tEnd / h / 150)),
        stopOnNonFinite: false,
      })
      // The paths as rows per kept step.
      const { times, values } = paths(tr)
      return { tr, cloud: { t: toFlat(times), rows: toRows(values) } }
    },
    [make, proc, h, n],
    { mode: 'release' },
  )
  const { tr, cloud } = sim.value
  const grid = useMemo(() => ({ a: proc.range[0], b: proc.range[1], n: 161 }), [proc])
  const fp = useMemo(() => {
    if (!showDensity) return null
    // Start the density as a narrow Gaussian at x₀ (the point mass smoothed over a few cells).
    const w = 0.03
    const dt = 0.005
    const ft = trace(
      densityEvolution({ drift: proc.drift, sigma: proc.sigma, grid, dt, tEnd: proc.tEnd }),
      { u0: (x) => Math.exp(-((x - proc.x0) ** 2) / (2 * w * w)) / (w * Math.sqrt(2 * Math.PI)) },
      100_000,
      { every: Math.max(1, Math.round(proc.tEnd / dt / 300)) },
    )
    return {
      x: toFlat(gridPoints(grid)),
      t: ft.steps.map((s) => s.time),
      u: ft.steps.map((s) => toFlat(s.u)),
      mass: ft.steps.map((s) => s.mass),
    }
  }, [proc, grid, showDensity])
  // For each kept step of the paths, the density frame nearest in time.
  const fpAt = useMemo(() => {
    if (!fp) return null
    let j = 0
    return Array.from(cloud.t, (t) => {
      while (j + 1 < fp.t.length && Math.abs(fp.t[j + 1] - t) <= Math.abs(fp.t[j] - t)) j++
      return j
    })
  }, [fp, cloud])

  const [at, setAt] = usePlayhead(cloud.t.length)
  const shown = Math.min(n, DRAWN)
  const growing = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i < shown; i++) {
      for (let k = 0; k <= at; k++) {
        x.push(cloud.t[k])
        y.push(cloud.rows[k][i])
      }
      x.push(NaN)
      y.push(NaN)
    }
    const now = cloud.rows[at].slice(0, shown)
    return { x, y, nowX: now.map(() => cloud.t[at]), nowY: now }
  }, [cloud, at, shown])
  const now = tr.steps[at]
  const values = cloud.rows[at]
  const mean = values.reduce((a, v) => a + v, 0) / values.length
  const sd = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length)
  const t = useAxis({ label: 't', range: [0, proc.tEnd] })
  const xAxis = useAxis({ label: 'X', range: proc.range })
  // The density axis: 1.2 × the larger peak at the final time, so the narrow start is clipped.
  const densityTop = useMemo(() => {
    const last = toFlat(histogram(Array.from(cloud.rows.at(-1)!), { bins: BINS, range: proc.range }).density)
    const peak = Math.max(...last, ...(fp ? fp.u.at(-1)! : []))
    return 1.2 * (Number.isFinite(peak) && peak > 0 ? peak : 1)
  }, [cloud, fp, proc])
  const dens = useAxis({ label: 'density', range: [0, densityTop] })
  return (
    <Figure
      title="Path clouds and their density"
      purpose="Many sample paths of an SDE drawn thin show its law spreading in time; the histogram of their values at each time matches the density from the Fokker–Planck equation."
      defaultSize="L"
      state={state}
      controls={
        <Player
          className="col-span-full"
          value={at}
          onChange={setAt}
          count={cloud.t.length}
          label="4 · time t"
          format={(i) => fmt(cloud.t[i])}
        />
      }
      readouts={{
        'at t': (
          <>
            <Readout label="steps so far" value={now.t} />
            <Readout label="mean, sd of X(t)" value={`${fmt(mean)}, ${fmt(sd)}`} />
            <Readout label="non-finite paths" value={now.nonFinite} />
            {fp && fpAt && <Readout label="density mass" value={fmt(fp.mass[fpAt[at]])} />}
          </>
        ),
      }}
      caption="Play to grow the paths from x₀. Left: up to 300 of the paths, each thin, drawn up to t, with their current values marked. Paths share keyed streams, so raising the count adds paths without changing the ones drawn. Right, on the same X axis: the histogram of all the paths' values at t, and the density at t from aifn/pde's conservative Fokker–Planck solver started from a narrow bump at x₀. The density axis is set by the final time, so the narrow start is clipped. In the double well the cloud splits between the two wells at ±1."
    >
      <Plots cols={2} widths={[2, 1]}>
        <Plot x={t} y={xAxis} legend={false}>
          <Curve name="paths" x={growing.x} y={growing.y} slot={0} thin silent live stale={sim.stale} />
          <Points name="now" x={growing.nowX} y={growing.nowY} slot={0} thin live />
        </Plot>
        <Plot x={dens} y={xAxis}>
          <Histogram name="histogram of X(t)" values={values} bins={BINS} range={proc.range} orient="y" slot={0} />
          {fp && fpAt && <Curve name="Fokker–Planck density" x={fp.u[fpAt[at]]} y={fp.x} slot={1} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Strong and weak convergence on geometric Brownian motion.

const STEPS = [0.2, 0.1, 0.05, 0.025, 0.0125]

export function ConvergenceSpecimen() {
  const state = useFigureState({
    process: row('1 · process', { sigma: slider(0.1, 1.5, 0.8, { label: 'volatility σ', step: 0.05 }) }),
  })
  const { sigma } = state.process
  // 3 schemes × 5 step sizes × 3000 paths: rerun on release.
  const computed = useComputed(
    () => {
      const gbm = geometricBrownianMotion({ mu: 1, sigma })
      const T = 1
      const opts = { x0: 1, paths: 3000 }
      return SCHEMES.map((sch, slot) => {
        const strong: number[] = []
        const weak: number[] = []
        for (const h of STEPS) {
          const approx = toFlat(
            run(sch.make(gbm.sde, { stepSize: h, tEnd: T }), opts, 1e5, { stream: stream('conv') }).x,
          )
          const exact = toFlat(run(gbm.exact({ stepSize: h, tEnd: T }), opts, 1e5, { stream: stream('conv') }).x)
          strong.push(approx.reduce((a, v, i) => a + Math.abs(v - exact[i]), 0) / approx.length)
          // The exact paths share the scheme's increments, so their mean cancels most of the Monte Carlo noise
          // (a control variate): E[X_h(1)] − E[X(1)] ≈ mean(X_h(1) − X(1)).
          weak.push(Math.abs(approx.reduce((a, v, i) => a + v - exact[i], 0) / approx.length))
        }
        return { ...sch, slot, strong, weak }
      })
    },
    [sigma],
    { mode: 'release' },
  )
  const results = computed.value
  const ref = (order: number, c: number) => STEPS.map((h) => c * h ** order)
  const half = ref(0.5, results[0].strong[0] / Math.sqrt(STEPS[0]))
  const one = ref(1, results[1].strong[0] / STEPS[0])
  const hx = useAxis({ label: 'h', log: true })
  const se = useAxis({ label: 'strong error', log: true })
  const we = useAxis({ label: 'weak error', log: true })
  const slope = (y: number[]) => Math.log(y[0] / y[y.length - 1]) / Math.log(STEPS[0] / STEPS[STEPS.length - 1])
  return (
    <Figure
      title="Strong and weak orders of convergence"
      purpose="Euler–Maruyama's paths converge at order ½ in h but its means at order 1; Milstein's Itô correction and Platen's derivative-free scheme raise the pathwise order to 1."
      state={state}
      readouts={{
        'fitted slopes': results.map(({ label, strong, weak }) => (
          <Readout key={label} label={label} value={`strong ${fmt(slope(strong))}, weak ${fmt(slope(weak))}`} />
        )),
      }}
      caption="Geometric Brownian motion dX = X dt + σX dW on [0, 1], 3000 paths. Strong error: mean |X_h(1) − X(1)| against the exact solution driven by the same Brownian increments. Weak error: |mean(X_h(1) − X(1))| over the same coupled paths, which estimates the bias E[X_h(1)] − e¹ with far less Monte Carlo noise than comparing the mean with e¹."
    >
      <Plots cols={2} hoverGroup>
        <Plot x={hx} y={se}>
          {results.map(({ label, slot, strong }) => (
            <Curve key={label} name={label} x={STEPS} y={strong} slot={slot} showPoints stale={computed.stale} />
          ))}
          <Curve name="slope ½" x={STEPS} y={half} dashed muted />
          <Curve name="slope 1" x={STEPS} y={one} dashed muted />
        </Plot>
        <Plot x={hx} y={we}>
          {results.map(({ label, slot, weak }) => (
            <Curve key={label} name={label} x={STEPS} y={weak} slot={slot} showPoints stale={computed.stale} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
