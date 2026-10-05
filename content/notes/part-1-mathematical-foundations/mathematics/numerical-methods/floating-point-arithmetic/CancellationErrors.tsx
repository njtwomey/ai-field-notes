import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type ExprId = 'cos' | 'exp' | 'sqrt'

/** Unit roundoff of IEEE binary64: half the gap between 1 and the next double. */
const U = 2 ** -53
const FLOOR = 1e-18

type Expr = {
  label: string
  naive: (x: number) => number
  stable: (x: number) => number
  /** Relative error predicted for the naive form: an O(u) error in a term of size 1, divided by the small result. */
  predicted: (x: number) => number
}

const EXPRESSIONS: Record<ExprId, Expr> = {
  cos: {
    label: '(1 − cos x)/x²',
    naive: (x) => (1 - Math.cos(x)) / (x * x),
    stable: (x) => (2 * Math.sin(x / 2) ** 2) / (x * x),
    predicted: (x) => (2 * U) / (x * x),
  },
  exp: {
    label: '(eˣ − 1)/x',
    naive: (x) => (Math.exp(x) - 1) / x,
    stable: (x) => Math.expm1(x) / x,
    predicted: (x) => U / x,
  },
  sqrt: {
    label: '√(1 + x) − 1',
    naive: (x) => Math.sqrt(1 + x) - 1,
    stable: (x) => x / (Math.sqrt(1 + x) + 1),
    predicted: (x) => (2 * U) / x,
  },
}

const LOG_X = toFlat(linspace(-10, 0, 301))

/**
 * Relative error of a naive and a rearranged formula, both evaluated in the browser's double precision. The
 * rearranged formula serves as the reference because it has no cancellation.
 */
export function CancellationErrors() {
  const state = useFigureState({
    id: choice<ExprId>(
      (Object.keys(EXPRESSIONS) as ExprId[]).map((k) => ({ value: k, label: EXPRESSIONS[k].label })),
      'cos',
      { label: 'expression' },
    ),
    logX: float(-6, { min: -10, max: 0, step: 0.5, label: 'log₁₀ x (readout)' }),
  })
  const e = EXPRESSIONS[state.id]

  const series = useMemo(() => {
    const rel = (f: (x: number) => number) =>
      LOG_X.map((lx) => {
        const x = 10 ** lx
        const truth = e.stable(x)
        return Math.max(FLOOR, Math.abs(f(x) - truth) / Math.abs(truth))
      })
    return [
      { name: `naive ${e.label}`, x: LOG_X, y: rel(e.naive), slot: 1 },
      {
        name: 'predicted: u × (size of cancelled terms / result)',
        x: LOG_X,
        y: LOG_X.map((lx) => Math.min(10, e.predicted(10 ** lx))),
        dashed: true,
        slot: 0,
      },
      { name: 'machine epsilon', x: [-10, 0], y: [2 * U, 2 * U], muted: true, dashed: true },
    ] as const
  }, [e])

  const x = 10 ** state.logX
  const naive = e.naive(x)
  const stable = e.stable(x)
  const xAxis = useAxis({ label: 'log₁₀ x', range: [-10, 0] })
  const yAxis = useAxis({ label: 'relative error', range: [1e-18, 10], log: true })
  return (
    <Figure
      title="Catastrophic cancellation in double precision"
      state={state}
      caption="Relative error of the naive formula, computed live in IEEE double precision, against an algebraically equal rearrangement that avoids subtracting nearly equal numbers. The naive error grows as x shrinks, as predicted by the rounding error in the large terms divided by the small result. For (1 − cos x)/x² the naive formula returns 0 below x ≈ 10⁻⁸."

      readouts={
        <>
          <Readout label="x" value={x.toExponential(1)} />
          <Readout label="naive" value={naive.toPrecision(17)} />
          <Readout label="rearranged" value={stable.toPrecision(17)} />
          <Readout label="relative error" value={formatNumber(Math.abs(naive - stable) / Math.abs(stable))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
