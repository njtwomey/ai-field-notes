import { Interactive, Readout, XYChart, formatNumber, useParam, type Handle, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const P = linspace(0.005, 1, 200)

/**
 * The penalty each score gives one case, as a function of the probability q assigned to the class that occurred: the
 * log score −log q is unbounded as q → 0; the (binary) Brier score (1 − q)² is at most 1.
 */
export function PenaltyCurves() {
  const q = useParam(0.1, { min: 0.005, max: 1, step: 0.005 })
  const series: XYSeries[] = [
    { name: 'log loss −ln q', type: 'line', x: P, y: P.map((p) => -Math.log(p)), slot: 0 },
    { name: 'Brier (1 − q)²', type: 'line', x: P, y: P.map((p) => (1 - p) ** 2), slot: 1 },
  ]
  const handles: Handle[] = [{ kind: 'x', at: q.value, label: 'q', onDrag: (x) => q.set(x) }]
  return (
    <Interactive
      title="How much one prediction costs"
      caption="Each curve is the penalty for one case as a function of q, the probability the model gave to the outcome that happened. Drag q towards 0: a confident wrong prediction costs a bounded 1 under the Brier score, and an unbounded amount under log loss."
      readout={
        <>
          <Readout label="q" value={formatNumber(q.value)} />
          <Readout label="log loss −ln q" value={formatNumber(-Math.log(q.value))} />
          <Readout label="Brier (1 − q)²" value={formatNumber((1 - q.value) ** 2)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart
          series={series}
          xLabel="probability q given to the true outcome"
          yLabel="penalty"
          xRange={[0, 1]}
          yRange={[0, 5]}
          handles={handles}
          height={300}
        />
      </div>
    </Interactive>
  )
}
