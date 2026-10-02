import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const R = 3.2
const ANGLES = linspace(0, 2 * Math.PI, 121)
const BLOW_UP = 1e3

/** Ellipse x^T Σ^{-1} x = c² for Σ = [[1, ρ], [ρ, 1]]. */
function ellipse(rho: number, c: number) {
  const a = c * Math.sqrt(1 + rho)
  const b = c * Math.sqrt(1 - rho)
  return {
    x: ANGLES.map((t) => (a * Math.cos(t) - b * Math.sin(t)) / Math.SQRT2),
    y: ANGLES.map((t) => (a * Math.cos(t) + b * Math.sin(t)) / Math.SQRT2),
  }
}

/**
 * HMC on a bivariate Gaussian with unit variances and correlation ρ, identity mass matrix. U(q) = q^T Σ^{-1} q / 2, so
 * ∇U = Σ^{-1} q. Each iteration draws a fresh momentum, runs L leapfrog steps of size ε and accepts with probability
 * min(1, exp(−ΔH)).
 */
function hmc(rho: number, eps: number, steps: number, iterations: number, start: [number, number], seed: number) {
  const det = 1 - rho * rho
  const grad = (x: number, y: number): [number, number] => [(x - rho * y) / det, (y - rho * x) / det]
  const U = (x: number, y: number) => (x * x - 2 * rho * x * y + y * y) / (2 * det)
  const g = rng(seed)
  let [qx, qy] = start
  const paths: { x: number[]; y: number[]; accepted: boolean }[] = []
  const deltas: number[] = []
  let acceptProb = 0
  for (let it = 0; it < iterations; it++) {
    let px = g.normal()
    let py = g.normal()
    const h0 = U(qx, qy) + (px * px + py * py) / 2
    let [x, y] = [qx, qy]
    const path = { x: [x], y: [y], accepted: false }
    let [gx, gy] = grad(x, y)
    let diverged = false
    for (let s = 0; s < steps; s++) {
      px -= (eps / 2) * gx
      py -= (eps / 2) * gy
      x += eps * px
      y += eps * py
      ;[gx, gy] = grad(x, y)
      px -= (eps / 2) * gx
      py -= (eps / 2) * gy
      path.x.push(x)
      path.y.push(y)
      if (Math.abs(x) > BLOW_UP || Math.abs(y) > BLOW_UP) {
        diverged = true
        break
      }
    }
    const dH = diverged ? Infinity : U(x, y) + (px * px + py * py) / 2 - h0
    const a = Math.min(1, Math.exp(-dH))
    acceptProb += a
    deltas.push(dH)
    if (g.uniform() < a) {
      qx = x
      qy = y
      path.accepted = true
    }
    paths.push(path)
  }
  return { paths, deltas, meanAccept: acceptProb / iterations, end: [qx, qy] as [number, number] }
}

export function HmcTrajectories() {
  const rho = useParam(0.9, { min: 0, max: 0.99, step: 0.01 })
  const eps = useParam(0.15, { min: 0.01, max: 1, step: 0.01 })
  const steps = useParam(20, { min: 1, max: 100, step: 1 })
  const iterations = useParam(5, { min: 1, max: 50, step: 1 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const sx = useParam(-2, { min: -R, max: R, step: 0.05 })
  const sy = useParam(-1.2, { min: -R, max: R, step: 0.05 })

  const run = useMemo(
    () => hmc(rho.value, eps.value, steps.value, iterations.value, [sx.value, sy.value], seed.value),
    [rho.value, eps.value, steps.value, iterations.value, sx.value, sy.value, seed.value],
  )

  const series: XYSeries[] = useMemo(() => {
    const e1 = ellipse(rho.value, 1)
    const e2 = ellipse(rho.value, 2)
    const out: XYSeries[] = [
      { name: '1 standard deviation', type: 'line', x: e1.x, y: e1.y, muted: true },
      { name: '2 standard deviations', type: 'line', x: e2.x, y: e2.y, muted: true },
    ]
    // One light series per trajectory, coloured by outcome; series with the same name share a legend entry.
    const many = run.paths.length > 1
    run.paths.forEach((p) =>
      out.push({
        name: p.accepted ? 'accepted trajectory' : 'rejected trajectory',
        type: 'line',
        x: p.x,
        y: p.y,
        slot: p.accepted ? 0 : 1,
        thin: many,
      }),
    )
    const states = run.paths.filter((p) => p.accepted).map((p) => [p.x[p.x.length - 1], p.y[p.y.length - 1]])
    out.push({
      name: 'accepted state',
      type: 'scatter',
      x: states.map((s) => s[0]),
      y: states.map((s) => s[1]),
      slot: 0,
    })
    return out
  }, [run, rho.value])

  const handles: Handle[] = [
    {
      kind: 'point',
      at: [sx.value, sy.value],
      label: 'start',
      onDrag: ([x, y]) => {
        sx.set(x)
        sy.set(y)
      },
    },
  ]
  const limit = 2 * Math.sqrt(1 - rho.value)
  const first = run.deltas[0]

  return (
    <Interactive
      title="Hamiltonian trajectories on a correlated Gaussian"
      caption="Each iteration draws a random momentum and follows L leapfrog steps of size ε along the Hamiltonian flow, then accepts the end point with probability min(1, e^(−ΔH)). The iterations slider sets how many trajectories the chain runs, each from a fresh momentum draw; with more than one they are drawn as light lines, and the accepted states are marked. Drag the start point. The trajectories sweep along the long axis of the ellipse, far beyond what a random walk reaches in one step. The energy error ΔH stays small for small ε; as ε approaches the stability limit 2σ_min, where σ_min = √(1 − ρ) is the narrowest standard deviation, ΔH grows and trajectories are rejected; above it they spiral out. Very long trajectories curve back on themselves, which is what NUTS detects."
      controls={
        <>
          <ParamSlider label="step size ε" param={eps} />
          <ParamSlider label="leapfrog steps L" param={steps} format={(v) => String(v)} />
          <ParamSlider label="correlation ρ" param={rho} />
          <ParamSlider label="iterations" param={iterations} format={(v) => String(v)} withArrows />
          <ParamSlider label="momentum seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="ΔH, first trajectory" value={Number.isFinite(first) ? formatNumber(first) : 'diverged'} />
          <Readout label="mean acceptance probability" value={formatNumber(run.meanAccept)} />
          <Readout label="trajectory length εL" value={formatNumber(eps.value * steps.value)} />
          <Readout label="stability limit 2σ_min" value={formatNumber(limit)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="q₁"
        yLabel="q₂"
        xRange={[-R, R]}
        yRange={[-R, R]}
        equalAspect
        handles={handles}
      />
    </Interactive>
  )
}
