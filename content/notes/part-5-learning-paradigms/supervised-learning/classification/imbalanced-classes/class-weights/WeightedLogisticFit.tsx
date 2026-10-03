import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'
import { expit } from '../_shared/binormal'

const N = 10000
const D = 2
const S = linspace(-4, 6, 201)

/** Weighted logistic regression on one feature by Newton's method. Returns [slope, intercept]. */
function fitWeighted(s: number[], y: number[], w: (label: number) => number): [number, number] {
  let a = 0
  let b = 0
  for (let iter = 0; iter < 30; iter++) {
    let ga = 0
    let gb = 0
    let haa = 0
    let hab = 0
    let hbb = 0
    for (let i = 0; i < s.length; i++) {
      const wi = w(y[i])
      const p = expit(a * s[i] + b)
      const r = wi * (p - y[i])
      const v = wi * p * (1 - p)
      ga += r * s[i]
      gb += r
      haa += v * s[i] * s[i]
      hab += v * s[i]
      hbb += v
    }
    // A tiny ridge keeps the step finite if the classes happen to be separable.
    haa += 1e-6
    hbb += 1e-6
    const det = haa * hbb - hab * hab
    const da = (hbb * ga - hab * gb) / det
    const db = (haa * gb - hab * ga) / det
    a -= da
    b -= db
    if (Math.abs(da) + Math.abs(db) < 1e-9) break
  }
  return [a, b]
}

/**
 * Logistic regression fitted with and without a weight w on the positive class. Under a well-specified model the
 * weight moves only the intercept, by log w; with unequal class variances the model is misspecified and the slope
 * moves too.
 */
export function WeightedLogisticFit() {
  const pi = useParam(0.05, { min: 0.01, max: 0.3, step: 0.01 })
  const w = useParam(19, { min: 1, max: 100, step: 1 })
  const [misspecified, setMisspecified] = useState(false)

  const sdPositive = misspecified ? 0.5 : 1
  const data = useMemo(() => {
    const g = rng(11)
    const s: number[] = []
    const y: number[] = []
    for (let i = 0; i < N; i++) {
      const positive = g.uniform() < pi.value
      y.push(positive ? 1 : 0)
      s.push(positive ? D + sdPositive * g.normal() : g.normal())
    }
    return { s, y }
  }, [pi.value, sdPositive])

  const plain = useMemo(() => fitWeighted(data.s, data.y, () => 1), [data])
  const weighted = useMemo(() => fitWeighted(data.s, data.y, (label) => (label === 1 ? w.value : 1)), [data, w.value])
  const shift = Math.log(w.value)

  const series = useMemo<XYSeries[]>(() => {
    const truth = S.map((v) => {
      const pos = pi.value * (normalPdf((v - D) / sdPositive) / sdPositive)
      const neg = (1 - pi.value) * normalPdf(v)
      return pos / (pos + neg)
    })
    return [
      { name: 'data', type: 'scatter', x: data.s, y: data.y.map((v) => (v === 1 ? 1.02 : -0.02)), muted: true },
      { name: 'true posterior', type: 'line', x: S, y: truth, dashed: true, muted: true },
      { name: 'unweighted fit', type: 'line', x: S, y: S.map((v) => expit(plain[0] * v + plain[1])), slot: 0 },
      { name: 'weighted fit', type: 'line', x: S, y: S.map((v) => expit(weighted[0] * v + weighted[1])), slot: 1 },
      {
        name: 'weighted, intercept − log w',
        type: 'line',
        x: S,
        y: S.map((v) => expit(weighted[0] * v + weighted[1] - shift)),
        slot: 2,
        dashed: true,
      },
    ]
  }, [data, plain, weighted, shift, pi.value, sdPositive])

  return (
    <Interactive
      title="Class weights move the intercept"
      caption={`${N} simulated scores: negatives N(0, 1), positives N(2, σ²). Logistic regression is fitted without weights and with weight w on each positive. With σ = 1 the true posterior is logistic, the two fits have the same slope, and subtracting log w from the weighted intercept recovers the unweighted curve. Switch on the misspecified model (σ = 0.5): the true posterior is no longer logistic, and the weight now changes the slope too.`}
      controls={
        <>
          <ParamSlider label="prevalence π" param={pi} />
          <ParamSlider label="weight on positives w" param={w} />
          <ParamSwitch label="misspecified (σ = 0.5)" checked={misspecified} onChange={setMisspecified} />
        </>
      }
      readout={
        <>
          <Readout label="positives" value={data.y.reduce((a, b) => a + b, 0)} />
          <Readout label="unweighted slope, intercept" value={`${formatNumber(plain[0])}, ${formatNumber(plain[1])}`} />
          <Readout
            label="weighted slope, intercept"
            value={`${formatNumber(weighted[0])}, ${formatNumber(weighted[1])}`}
          />
          <Readout label="intercept change" value={formatNumber(weighted[1] - plain[1])} />
          <Readout label="log w" value={formatNumber(shift)} />
        </>
      }
    >
      <XYChart
        height={340}
        xLabel="score s"
        yLabel="P(y = 1 | s)"
        xRange={[-4, 6]}
        yRange={[-0.05, 1.05]}
        series={series}
      />
    </Interactive>
  )
}
