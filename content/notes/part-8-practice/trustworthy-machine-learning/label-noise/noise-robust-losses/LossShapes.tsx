import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

const P = linspace(0.01, 1, 200)
const VIEW = [
  { value: 'loss', label: 'loss' },
  { value: 'weight', label: 'gradient weight' },
] as const
type View = (typeof VIEW)[number]['value']

const gce = (p: number, q: number) => (1 - p ** q) / q

/**
 * Cross-entropy, MAE and generalised cross-entropy as functions of the probability p assigned to the given label, and
 * the weight each puts on −∇p relative to cross-entropy's 1/p.
 */
export function LossShapes() {
  const q = useParam(0.7, { min: 0.05, max: 1, step: 0.05 })
  const p = useParam(0.05, { min: 0.01, max: 1, step: 0.01 })
  const [view, setView] = useState<View>('loss')

  const series: XYSeries[] = useMemo(() => {
    const f =
      view === 'loss'
        ? { ce: (x: number) => -Math.log(x), mae: (x: number) => 2 * (1 - x), gce: (x: number) => gce(x, q.value) }
        : { ce: (x: number) => 1 / x, mae: () => 2, gce: (x: number) => x ** (q.value - 1) }
    return [
      { name: 'cross-entropy', type: 'line', x: P, y: P.map(f.ce), slot: 0 },
      { name: 'MAE', type: 'line', x: P, y: P.map(f.mae), slot: 1 },
      { name: `GCE, q = ${formatNumber(q.value)}`, type: 'line', x: P, y: P.map(f.gce), slot: 2 },
    ]
  }, [view, q.value])

  return (
    <Interactive
      title="How much a doubtful label pulls"
      caption="The horizontal axis is the probability p the model gives to the observed label. A mislabelled example that the model fits well under its true class has small p. Cross-entropy's loss and gradient weight 1/p grow without bound there, so such examples dominate training. MAE weights every example equally and generalised cross-entropy (GCE) weights by p to the power q − 1, between the two. Drag the vertical line to read off values at one p."
      controls={
        <>
          <ParamChoice label="show" value={view} onChange={setView} options={VIEW} />
          <ParamSlider label="GCE exponent q" param={q} />
          <ParamSlider label="p for readout" param={p} />
        </>
      }
      readout={
        <>
          <Readout label="cross-entropy" value={formatNumber(view === 'loss' ? -Math.log(p.value) : 1 / p.value)} />
          <Readout label="MAE" value={formatNumber(view === 'loss' ? 2 * (1 - p.value) : 2)} />
          <Readout
            label="GCE"
            value={formatNumber(view === 'loss' ? gce(p.value, q.value) : p.value ** (q.value - 1))}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="probability of the observed label, p"
        yLabel={view === 'loss' ? 'loss' : 'weight on −∇p'}
        xRange={[0, 1]}
        yRange={[0, view === 'loss' ? 5 : 25]}
        handles={[{ kind: 'x', at: p.value, onDrag: p.set, label: 'p' }]}
        ariaLabel="Loss or gradient weight against the probability of the observed label"
      />
    </Interactive>
  )
}
