import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

const N = 200
const EXPERIMENTS = 400
const BINS = 31

type Arm = { x: number[]; y: number[] }

/** One arm of standardised users: pre-period X ~ N(0, 1), post-period Y = ρX + √(1 − ρ²)ε + effect. */
function drawArm(g: ReturnType<typeof rng>, rho: number, effect: number): Arm {
  const x: number[] = []
  const y: number[] = []
  const s = Math.sqrt(1 - rho * rho)
  for (let i = 0; i < N; i++) {
    const xi = g.normal()
    x.push(xi)
    y.push(rho * xi + s * g.normal() + effect)
  }
  return { x, y }
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

/** Raw difference in means and the CUPED estimate with θ = cov(X, Y)/var(X) pooled over both arms. */
function estimate(t: Arm, c: Arm) {
  const x = [...t.x, ...c.x]
  const y = [...t.y, ...c.y]
  const mx = mean(x)
  const my = mean(y)
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < x.length; i++) {
    sxy += (x[i] - mx) * (y[i] - my)
    sxx += (x[i] - mx) ** 2
  }
  const theta = sxy / sxx
  const raw = mean(t.y) - mean(c.y)
  return { theta, raw, cuped: raw - theta * (mean(t.x) - mean(c.x)) }
}

function histogram(values: number[], lo: number, hi: number) {
  const width = (hi - lo) / BINS
  const counts = new Array<number>(BINS).fill(0)
  for (const v of values) {
    const b = Math.floor((v - lo) / width)
    if (b >= 0 && b < BINS) counts[b]++
  }
  return {
    x: counts.map((_, i) => lo + (i + 0.5) * width),
    y: counts.map((n) => n / (values.length * width)),
  }
}

const sd = (v: number[]) => {
  const m = mean(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
}

/**
 * CUPED on simulated experiments with 200 users per arm. Left: one experiment's pre-period covariate against the
 * outcome, with a line of slope θ through each arm's means; the CUPED estimate is the vertical gap between the lines.
 * Right: the raw and CUPED estimates over 400 experiments.
 */
export function CupedDemo() {
  const [rho, setRho] = useState(0.7)
  const [effect, setEffect] = useState(0.1)
  const [seed, setSeed] = useState(1)

  const r = useMemo(() => {
    const g = rng(seed * 7919 + Math.round(rho * 1000))
    const shown = { t: drawArm(g, rho, effect), c: drawArm(g, rho, 0) }
    const one = estimate(shown.t, shown.c)
    const raws: number[] = []
    const cupeds: number[] = []
    for (let e = 0; e < EXPERIMENTS; e++) {
      const est = estimate(drawArm(g, rho, effect), drawArm(g, rho, 0))
      raws.push(est.raw)
      cupeds.push(est.cuped)
    }
    const spread = 4.5 * Math.sqrt(2 / N)
    const hr = histogram(raws, effect - spread, effect + spread)
    const hc = histogram(cupeds, effect - spread, effect + spread)
    const lineAt = (arm: Arm) => {
      const mx = mean(arm.x)
      const my = mean(arm.y)
      return { x: [-3, 3], y: [my + one.theta * (-3 - mx), my + one.theta * (3 - mx)] }
    }
    const scatter: XYSeries[] = [
      {
        name: 'users',
        type: 'scatter',
        x: [...shown.c.x, ...shown.t.x],
        y: [...shown.c.y, ...shown.t.y],
        group: [...shown.c.x.map(() => 0), ...shown.t.x.map(() => 1)],
        groupNames: ['control', 'treatment'],
      },
      { name: 'control, slope θ', type: 'line', ...lineAt(shown.c), slot: 0 },
      { name: 'treatment, slope θ', type: 'line', ...lineAt(shown.t), slot: 1 },
    ]
    const peak = Math.max(...hc.y, ...hr.y)
    const hist: XYSeries[] = [
      { name: 'raw difference', type: 'line', x: hr.x, y: hr.y, slot: 2 },
      { name: 'CUPED', type: 'line', x: hc.x, y: hc.y, slot: 3 },
      { name: 'true effect', type: 'line', x: [effect, effect], y: [0, peak * 1.05], emphasis: true, dashed: true },
    ]
    return { one, scatter, hist, sdRaw: sd(raws), sdCuped: sd(cupeds) }
  }, [rho, effect, seed])

  return (
    <Interactive
      title="CUPED removes the variance explained by the pre-period"
      caption="Left: one simulated experiment with 200 users per arm. Each point is a user's pre-experiment metric X and in-experiment metric Y, both standardised. The raw estimate is the difference in mean Y. CUPED draws a line of slope θ = cov(X, Y)/var(X) through each arm's means and reports the vertical gap between them, which corrects for one arm happening to contain heavier users. Right: the two estimators over 400 experiments. Both are centred on the true effect; the CUPED estimates are narrower by the factor √(1 − ρ²)."
      controls={
        <>
          <ParamSlider label="correlation ρ of X and Y" value={rho} onChange={setRho} min={0} max={0.95} step={0.05} />
          <ParamSlider
            label="true effect (sd units)"
            value={effect}
            onChange={setEffect}
            min={0}
            max={0.3}
            step={0.01}
          />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={20} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="this experiment: raw" value={formatNumber(r.one.raw)} />
          <Readout label="this experiment: CUPED" value={formatNumber(r.one.cuped)} />
          <Readout label="sd ratio, simulated" value={formatNumber(r.sdCuped / r.sdRaw)} />
          <Readout label="√(1 − ρ²)" value={formatNumber(Math.sqrt(1 - rho * rho))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          series={r.scatter}
          xRange={[-3, 3]}
          yRange={[-3.5, 3.5]}
          xLabel="pre-experiment X"
          yLabel="outcome Y"
        />
        <XYChart
          height={320}
          series={r.hist}
          yRange={[0, undefined]}
          xLabel="estimated effect"
          yLabel="density over experiments"
        />
      </div>
    </Interactive>
  )
}
