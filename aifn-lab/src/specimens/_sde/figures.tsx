import { gridPoints } from 'aifn/pde'
import { stream } from 'aifn/random'
import {
  densityEvolution,
  eulerMaruyama,
  geometricBrownianMotion,
  milstein,
  ornsteinUhlenbeck,
  paths,
  stochasticRungeKutta,
  type Sde,
  type SdeOptions,
  type SdeState,
} from 'aifn/sde'
import { histogram } from 'aifn/stats'
import { mul, sub, toFlat, toRows } from 'aifn/tensor'
import { run, trace, type Algorithm } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, Switch, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type XYSeries } from '@lab/viz'

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
const COUNTS = [10, 50, 200, 1000, 3000].map((n) => ({ value: String(n), label: `${n} paths` }))

const NONE: XYSeries[] = []
const BINS = 40
const DRAWN = 300

/** A histogram as a sideways step outline: x is the density, y runs through the bin edges. */
function sidewaysHistogram(values: ArrayLike<number>, range: [number, number]) {
  const h = histogram(Array.from(values), { bins: BINS, range })
  const e = h.edges
  const x: number[] = [0]
  const y: number[] = [e[0]]
  for (let i = 0; i < BINS; i++) {
    x.push(h.density[i], h.density[i])
    y.push(e[i], e[i + 1])
  }
  x.push(0)
  y.push(e[BINS])
  return { x, y }
}

export function PathCloudSpecimen() {
  const [which, setWhich] = useState('double-well')
  const [scheme, setScheme] = useState<SchemeName>('euler-maruyama')
  const [count, setCount] = useState('200')
  const [h, setH] = useState(0.02)
  const [showDensity, setShowDensity] = useState(true)
  const proc = PROCESSES.find((p) => p.value === which)!
  const make = SCHEMES.find((s) => s.value === scheme)!.make
  const n = Number(count)
  const tr = useMemo(
    () =>
      trace(make(proc.sde, { h, tEnd: proc.tEnd }), { x0: proc.x0, paths: n }, 100_000, {
        stream: stream('path-cloud'),
        every: Math.max(1, Math.round(proc.tEnd / h / 150)),
        stopOnNonFinite: false,
      }),
    [make, proc, h, n],
  )
  // The paths as rows per kept step, and the sideways histogram of the cloud at every kept step.
  const cloud = useMemo(() => {
    const { times, values } = paths(tr)
    const rows = toRows(values)
    return { t: toFlat(times), rows, hist: rows.map((r) => sidewaysHistogram(r, proc.range)) }
  }, [tr, proc])
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
      t: ft.steps.map((s) => s.t),
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
  // The density axis: 1.2 × the larger peak at the final time, so the narrow start is clipped.
  const densityTop = useMemo(() => {
    const hist = cloud.hist.at(-1)!
    const peak = Math.max(...hist.x, ...(fp ? fp.u.at(-1)! : []))
    return 1.2 * (Number.isFinite(peak) && peak > 0 ? peak : 1)
  }, [cloud, fp])

  const [at, setAt] = usePlayhead(cloud.t.length)
  const shown = Math.min(n, DRAWN)
  const growing = useMemo((): XYSeries[] => {
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
    return [
      { name: 'paths', type: 'line', thin: true, slot: 0, x, y },
      { name: 'now', type: 'scatter', slot: 0, x: now.map(() => cloud.t[at]), y: now },
    ]
  }, [cloud, at, shown])
  const side: XYSeries[] = [
    { name: `histogram of X(t)`, type: 'line', slot: 0, ...cloud.hist[at] },
    ...(fp && fpAt
      ? [{ name: 'Fokker–Planck density', type: 'line' as const, slot: 1, x: fp.u[fpAt[at]], y: fp.x }]
      : []),
  ]
  const now = tr.steps[at]
  const values = cloud.rows[at]
  const mean = values.reduce((a, v) => a + v, 0) / values.length
  const sd = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length)
  return (
    <Figure
      title="Path clouds and their density"
      description="Many sample paths of an SDE drawn thin show its law spreading in time; the histogram of their values at each time matches the density from the Fokker–Planck equation."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · process">
            <Select label="SDE" value={which} onChange={setWhich} options={PROCESSES} />
          </ControlRow>
          <ControlRow label="2 · simulation">
            <Select label="scheme" value={scheme} onChange={setScheme} options={SCHEMES} />
            <Select label="paths" value={count} onChange={setCount} options={COUNTS} />
            <Slider label="step h" value={h} min={0.002} max={0.2} step={0.002} onChange={setH} />
          </ControlRow>
          <ControlRow label="3 · reveal">
            <Switch label="Fokker–Planck density" checked={showDensity} onChange={setShowDensity} />
          </ControlRow>
          <ControlRow label="4 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={cloud.t.length}
              duration={5}
              label="t"
              format={(i) => fmt(cloud.t[i])}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="steps so far" value={now.step} />
          <Readout label="mean, sd of X(t)" value={`${fmt(mean)}, ${fmt(sd)}`} />
          <Readout label="non-finite paths" value={now.nonFinite} />
          {fp && fpAt && <Readout label="density mass" value={fmt(fp.mass[fpAt[at]])} />}
        </>
      }
      caption="Play to grow the paths from x₀. Left: up to 300 of the paths, each thin, drawn up to t, with their current values marked. Paths share keyed streams, so raising the count adds paths without changing the ones drawn. Right, on the same X axis: the histogram of all the paths' values at t, and the density at t from aifn/pde's conservative Fokker–Planck solver started from a narrow bump at x₀. The density axis is set by the final time, so the narrow start is clipped. In the double well the cloud splits between the two wells at ±1."
    >
      <Subplots cols={2} widthRatios={[2, 1]} sharey>
        <Panel>
          <XYChart
            series={NONE}
            live={growing}
            xLabel="t"
            yLabel="X"
            xRange={[0, proc.tEnd]}
            yRange={proc.range}
            legend={false}
          />
        </Panel>
        <Panel>
          <XYChart series={NONE} live={side} xLabel="density" yLabel="X" xRange={[0, densityTop]} yRange={proc.range} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Strong and weak convergence on geometric Brownian motion.

const STEPS = [0.2, 0.1, 0.05, 0.025, 0.0125]

export function ConvergenceSpecimen() {
  const [sigma, setSigma] = useState(0.8)
  const results = useMemo(() => {
    const gbm = geometricBrownianMotion({ mu: 1, sigma })
    const T = 1
    const opts = { x0: 1, paths: 3000 }
    return SCHEMES.map((sch, slot) => {
      const strong: number[] = []
      const weak: number[] = []
      for (const h of STEPS) {
        const approx = toFlat(run(sch.make(gbm.sde, { h, tEnd: T }), opts, 1e5, { stream: stream('conv') }).x)
        const exact = toFlat(run(gbm.exact({ h, tEnd: T }), opts, 1e5, { stream: stream('conv') }).x)
        strong.push(approx.reduce((a, v, i) => a + Math.abs(v - exact[i]), 0) / approx.length)
        // The exact paths share the scheme's increments, so their mean cancels most of the Monte Carlo noise
        // (a control variate): E[X_h(1)] − E[X(1)] ≈ mean(X_h(1) − X(1)).
        weak.push(Math.abs(approx.reduce((a, v, i) => a + v - exact[i], 0) / approx.length))
      }
      return { ...sch, slot, strong, weak }
    })
  }, [sigma])
  const ref = (order: number, c: number) => STEPS.map((h) => c * h ** order)
  const strongSeries: XYSeries[] = [
    ...results.map(({ label, slot, strong }) => ({
      name: label,
      slot,
      type: 'line' as const,
      showPoints: true,
      x: STEPS,
      y: strong,
    })),
    {
      name: 'slope ½',
      type: 'line',
      dashed: true,
      muted: true,
      x: STEPS,
      y: ref(0.5, results[0].strong[0] / Math.sqrt(STEPS[0])),
    },
    { name: 'slope 1', type: 'line', dashed: true, muted: true, x: STEPS, y: ref(1, results[1].strong[0] / STEPS[0]) },
  ]
  const weakSeries: XYSeries[] = results.map(({ label, slot, weak }) => ({
    name: label,
    slot,
    type: 'line',
    showPoints: true,
    x: STEPS,
    y: weak,
  }))
  const slope = (y: number[]) => Math.log(y[0] / y[y.length - 1]) / Math.log(STEPS[0] / STEPS[STEPS.length - 1])
  return (
    <Figure
      title="Strong and weak orders of convergence"
      description="Euler–Maruyama's paths converge at order ½ in h but its means at order 1; Milstein's Itô correction and Platen's derivative-free scheme raise the pathwise order to 1."
      controls={<Slider label="volatility σ" value={sigma} min={0.1} max={1.5} step={0.05} onChange={setSigma} />}
      readouts={results.map(({ label, strong, weak }) => (
        <Readout key={label} label={label} value={`strong ${fmt(slope(strong))}, weak ${fmt(slope(weak))}`} />
      ))}
      caption="Geometric Brownian motion dX = X dt + σX dW on [0, 1], 3000 paths. Strong error: mean |X_h(1) − X(1)| against the exact solution driven by the same Brownian increments. Weak error: |mean(X_h(1) − X(1))| over the same coupled paths, which estimates the bias E[X_h(1)] − e¹ with far less Monte Carlo noise than comparing the mean with e¹."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={strongSeries} xLog yLog xLabel="h" yLabel="strong error" />
        </Panel>
        <Panel>
          <XYChart series={weakSeries} xLog yLog xLabel="h" yLabel="weak error" />
        </Panel>
      </Subplots>
    </Figure>
  )
}
