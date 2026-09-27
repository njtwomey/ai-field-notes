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
} from '@/components/viz'
import { linspace } from '@/lib/math'

const ANGLES = linspace(0, 180, 181)
const X_RANGE: [number, number] = [0, 180]
const Y_RANGE: [number, number | undefined] = [0, 1]

/** Probability that a pair at angle θ (degrees) shares a bucket in at least one of L tables of k random-hyperplane bits. */
const retrieve = (theta: number, k: number, L: number) => 1 - (1 - (1 - theta / 180) ** k) ** L

/** The S-curve of random-hyperplane LSH: AND over k bits, OR over L tables. */
export function LshSCurve() {
  const k = useParam(10, { min: 1, max: 30, step: 1 })
  const L = useParam(20, { min: 1, max: 100, step: 1 })
  const near = useParam(30, { min: 0, max: 180, step: 1 })
  const far = useParam(90, { min: 0, max: 180, step: 1 })

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'one hash', type: 'line', x: ANGLES, y: ANGLES.map((t) => 1 - t / 180), muted: true, dashed: true },
      {
        name: `k = ${k.value}, L = 1`,
        type: 'line',
        x: ANGLES,
        y: ANGLES.map((t) => retrieve(t, k.value, 1)),
        slot: 1,
      },
      {
        name: `k = ${k.value}, L = ${L.value}`,
        type: 'line',
        x: ANGLES,
        y: ANGLES.map((t) => retrieve(t, k.value, L.value)),
        slot: 0,
      },
    ],
    [k.value, L.value],
  )
  const handles: Handle[] = [
    { kind: 'x', at: near.value, label: 'near', onDrag: (x) => near.set(x) },
    { kind: 'x', at: far.value, label: 'far', onDrag: (x) => far.set(x) },
  ]
  const p1 = 1 - near.value / 180
  const p2 = 1 - far.value / 180
  const rho = p1 > p2 && p2 > 0 && p1 < 1 ? Math.log(1 / p1) / Math.log(1 / p2) : NaN
  // Where p^k = 1/L the retrieval probability is 1 - (1 - 1/L)^L, at least 1 - 1/e: the knee of the S-curve.
  const threshold = 180 * (1 - (1 / L.value) ** (1 / k.value))

  return (
    <Interactive
      title="Locality-sensitive hashing: amplification by AND and OR"
      caption="Random-hyperplane hashing: one bit collides for a pair at angle θ with probability 1 − θ/180°. A key of k bits (AND) collides with probability (1 − θ/180°)^k; L independent tables (OR) retrieve the pair with probability 1 − (1 − (1 − θ/180°)^k)^L. Raising k moves the S-curve left and makes it steeper; raising L moves it right. Drag the near and far lines to set the angles a query must separate."
      controls={
        <>
          <ParamSlider label="bits per key k" param={k} format={(v) => String(v)} withArrows />
          <ParamSlider label="tables L" param={L} format={(v) => String(v)} withArrows />
          <ParamSlider label="near angle (°)" param={near} format={(v) => String(v)} />
          <ParamSlider label="far angle (°)" param={far} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="P(retrieve near)" value={formatNumber(retrieve(near.value, k.value, L.value))} />
          <Readout label="P(retrieve far)" value={formatNumber(retrieve(far.value, k.value, L.value))} />
          <Readout label="ρ = ln(1/p₁)/ln(1/p₂)" value={Number.isFinite(rho) ? formatNumber(rho) : '–'} />
          <Readout label="knee where p^k = 1/L (°)" value={formatNumber(threshold)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="angle θ between query and point (degrees)"
        yLabel="probability of retrieval"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        handles={handles}
        height={340}
      />
    </Interactive>
  )
}
