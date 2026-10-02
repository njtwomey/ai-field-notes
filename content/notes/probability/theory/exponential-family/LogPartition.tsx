import { useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

type FamilyId = 'bernoulli' | 'poisson' | 'exponential'

type Family = {
  label: string
  /** Natural parameter range shown on the axis. */
  range: [number, number]
  initial: number
  A: (eta: number) => number
  /** A′(η) = E[T(X)]. */
  mean: (eta: number) => number
  /** A″(η) = var[T(X)]. */
  variance: (eta: number) => number
  /** The usual parameter, as text. */
  usual: (eta: number) => string
}

const FAMILIES: Record<FamilyId, Family> = {
  bernoulli: {
    label: 'Bernoulli',
    range: [-6, 6],
    initial: 1,
    A: (eta) => Math.log1p(Math.exp(eta)),
    mean: (eta) => 1 / (1 + Math.exp(-eta)),
    variance: (eta) => {
      const p = 1 / (1 + Math.exp(-eta))
      return p * (1 - p)
    },
    usual: (eta) => `p = ${formatNumber(1 / (1 + Math.exp(-eta)))}`,
  },
  poisson: {
    label: 'Poisson',
    range: [-3, 3],
    initial: 1,
    A: (eta) => Math.exp(eta),
    mean: (eta) => Math.exp(eta),
    variance: (eta) => Math.exp(eta),
    usual: (eta) => `λ = ${formatNumber(Math.exp(eta))}`,
  },
  exponential: {
    label: 'exponential',
    range: [-5, -0.2],
    initial: -1,
    A: (eta) => -Math.log(-eta),
    mean: (eta) => -1 / eta,
    variance: (eta) => 1 / (eta * eta),
    usual: (eta) => `λ = ${formatNumber(-eta)}`,
  },
}

/** The log-partition function A(η) of a one-parameter family, with its tangent: the slope is the mean of T(X). */
export function LogPartition() {
  const [id, setId] = useState<FamilyId>('bernoulli')
  // One η per family, so switching family keeps each family's own value inside its own range.
  const [etas, setEtas] = useState(() => ({
    bernoulli: FAMILIES.bernoulli.initial,
    poisson: FAMILIES.poisson.initial,
    exponential: FAMILIES.exponential.initial,
  }))
  const f = FAMILIES[id]
  const [lo, hi] = f.range
  const setEta = (v: number) =>
    setEtas((prev) => ({ ...prev, [id]: Number((Math.round(Math.min(Math.max(v, lo), hi) * 20) / 20).toFixed(2)) }))

  const xs = linspace(lo, hi, 200)
  const e = etas[id]
  const slope = f.mean(e)
  const series: XYSeries[] = [
    { name: 'A(η)', type: 'line', x: xs, y: xs.map(f.A), slot: 0 },
    {
      name: 'tangent, slope E[T(X)]',
      type: 'line',
      x: xs,
      y: xs.map((x) => f.A(e) + slope * (x - e)),
      dashed: true,
      slot: 1,
    },
  ]
  const ys = xs.map(f.A)
  const yRange: [number, number] = [Math.min(...ys) - 0.5, Math.max(...ys) + 0.5]
  const handles: Handle[] = [{ kind: 'point', at: [e, f.A(e)], onDrag: ([x]) => setEta(x) }]

  return (
    <Interactive
      title="The log-partition function generates the moments"
      caption="A(η) is convex. Its slope at η is the mean of the sufficient statistic, and its curvature is the variance. Drag the point along the curve, or use the slider. For the Poisson, A = A′ = A″ = e^η, so the mean equals the variance."
      controls={
        <>
          <ParamChoice
            label="family"
            value={id}
            onChange={setId}
            options={(Object.keys(FAMILIES) as FamilyId[]).map((k) => ({ value: k, label: FAMILIES[k].label }))}
          />
          <ParamSlider label="natural parameter η" value={e} onChange={setEta} min={lo} max={hi} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="usual parameter" value={f.usual(e)} />
          <Readout label="A′(η) = E[T(X)]" value={formatNumber(slope)} />
          <Readout label="A″(η) = var[T(X)]" value={formatNumber(f.variance(e))} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="η"
        yLabel="A(η)"
        series={series}
        xRange={f.range}
        yRange={yRange}
        handles={handles}
      />
    </Interactive>
  )
}
