import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  NumberField,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bsplineRow, evaluate, makeBasis, penalise, smooth } from './splines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 80
const NOISE = 0.3
const GRID = toFlat(linspace(0, 1, 161))
/** The true curve oscillates faster as x grows, so evenly spaced knots are not the best placement. */
const truth = (x: number) => Math.sin(4 * Math.PI * x * x)
const even = (k: number) => Array.from({ length: k }, (_, i) => (i + 1) / (k + 1))

type Props = {
  /** Number of interior knots at the start, evenly spaced. */
  initialKnots?: number
  /** Knots can be dragged and their number changed. Otherwise they stay evenly spaced. */
  draggable?: boolean
  /** Starting log₁₀ λ. The slider runs from −9 (no penalty to speak of) to 1 (a straight line). */
  initialLogLambda?: number
}

export function SplineExplorer({ initialKnots = 4, draggable = true, initialLogLambda = -9 }: Props) {
  const [knots, setKnots] = useState(() => even(initialKnots))
  const state = useFigureState({
    logLambda: float(initialLogLambda, { min: -9, max: 1, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    showBasis: setting(false, 'show scaled basis functions'),
    seed: int(4, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    return { x, y: x.map((xi) => truth(xi) + NOISE * normal(r)) }
  }, [state.seed])

  const basis0 = useMemo(() => makeBasis(data.x, knots, 0, 1), [data, knots])
  const smoother = useMemo(() => penalise(basis0, 10 ** state.logLambda), [basis0, state.logLambda])
  const fit = useMemo(() => smooth(smoother, data.y), [smoother, data])
  const curve = useMemo(() => evaluate(smoother, fit.coef, GRID), [smoother, fit])
  const rss = data.y.reduce((s, y, i) => s + (y - fit.fitted[i]) ** 2, 0)
  const gcv = (N * rss) / (N - smoother.edf) ** 2

  // Each basis function scaled by its coefficient: the fit is their sum. Drawn as muted segments.
  const basis: Segment[] = useMemo(() => {
    if (!state.showBasis) return []
    const rows = GRID.map((g) => bsplineRow(g, smoother.knots, smoother.degree))
    return fit.coef.flatMap((c, j) =>
      GRID.slice(1).map((g, i) => ({ from: [GRID[i], c * rows[i][j]], to: [g, c * rows[i + 1][j]] }) as Segment),
    )
  }, [state.showBasis, smoother, fit])

  const handles: Handle[] = draggable
    ? knots.map((k, i) => ({
        kind: 'x',
        at: k,
        label: 'knot',
        onDrag: (x: number) => setKnots((ks) => ks.map((v, j) => (j === i ? Math.min(Math.max(x, 0.02), 0.98) : v))),
      }))
    : []

  const series = [
    { name: 'data', x: data.x, y: data.y, slot: 0 },
    { name: 'true curve', x: GRID, y: GRID.map(truth), slot: 2, dashed: true },
    { name: 'spline fit', x: GRID, y: curve, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2, 2] })
  return (
    <Figure
      title={draggable ? 'Knots and smoothing' : 'Smoothing parameter and effective degrees of freedom'}
      state={state}
      caption={
        draggable ? (
          <>
            A cubic B-spline fitted by penalised least squares. Drag the vertical knot lines, or change their number.
            The true curve oscillates faster on the right, so moving knots there improves the fit with the same number
            of parameters. Raising λ penalises curvature and lowers the effective degrees of freedom below the basis
            size; the largest λ leaves a straight line, 2 degrees of freedom.
          </>
        ) : (
          <>
            Twenty evenly spaced knots, more than the data need, with a curvature penalty λ. Small λ interpolates the
            noise; large λ flattens the fit towards a straight line. The effective degrees of freedom, the trace of the
            smoother matrix, falls continuously from the basis size to 2. Generalised cross-validation (GCV) is smallest
            at an intermediate λ.
          </>
        )
      }
      controls={
        draggable && (
          <NumberField
            label="interior knots"
            type="int"
            value={knots.length}
            onChange={(k) => setKnots(even(k))}
            min={1}
            max={10}
            suggestions={[1, 3, 5, 10]}
          />
        )
      }
      readouts={
        <>
          <Readout label="basis size" value={String(smoother.A.length)} />
          <Readout label="effective df" value={formatNumber(smoother.edf)} />
          <Readout label="RSS" value={formatNumber(rss)} />
          <Readout label="GCV" value={formatNumber(gcv)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Segments segments={basis} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
