import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { bsplineRow, evaluate, makeBasis, penalise, smooth } from './splines'

const N = 80
const NOISE = 0.3
const GRID = linspace(0, 1, 161)
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
  const [logLambda, setLogLambda] = useState(initialLogLambda)
  const [seed, setSeed] = useState(4)
  const [showBasis, setShowBasis] = useState(false)

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    return { x, y: x.map((xi) => truth(xi) + NOISE * r.normal()) }
  }, [seed])

  const basis0 = useMemo(() => makeBasis(data.x, knots, 0, 1), [data, knots])
  const smoother = useMemo(() => penalise(basis0, 10 ** logLambda), [basis0, logLambda])
  const fit = useMemo(() => smooth(smoother, data.y), [smoother, data])
  const curve = useMemo(() => evaluate(smoother, fit.coef, GRID), [smoother, fit])
  const rss = data.y.reduce((s, y, i) => s + (y - fit.fitted[i]) ** 2, 0)
  const gcv = (N * rss) / (N - smoother.edf) ** 2

  // Each basis function scaled by its coefficient: the fit is their sum. Drawn as muted segments.
  const basis: Segment[] = useMemo(() => {
    if (!showBasis) return []
    const rows = GRID.map((g) => bsplineRow(g, smoother.knots, smoother.degree))
    return fit.coef.flatMap((c, j) =>
      GRID.slice(1).map((g, i) => ({ from: [GRID[i], c * rows[i][j]], to: [g, c * rows[i + 1][j]] }) as Segment),
    )
  }, [showBasis, smoother, fit])

  const handles: Handle[] = draggable
    ? knots.map((k, i) => ({
        kind: 'x',
        at: k,
        label: 'knot',
        onDrag: (x: number) => setKnots((ks) => ks.map((v, j) => (j === i ? Math.min(Math.max(x, 0.02), 0.98) : v))),
      }))
    : []

  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, slot: 0 },
    { name: 'true curve', type: 'line', x: GRID, y: GRID.map(truth), slot: 2, dashed: true },
    { name: 'spline fit', type: 'line', x: GRID, y: curve, slot: 1 },
  ]

  return (
    <Interactive
      title={draggable ? 'Knots and smoothing' : 'Smoothing parameter and effective degrees of freedom'}
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
        <>
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-9}
            max={1}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          {draggable && (
            <ParamSlider
              label="interior knots"
              value={knots.length}
              onChange={(k) => setKnots(even(k))}
              min={1}
              max={10}
              step={1}
              withArrows
            />
          )}
          <ParamSwitch label="show scaled basis functions" checked={showBasis} onChange={setShowBasis} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="basis size" value={String(smoother.A.length)} />
          <Readout label="effective df" value={formatNumber(smoother.edf)} />
          <Readout label="RSS" value={formatNumber(rss)} />
          <Readout label="GCV" value={formatNumber(gcv)} />
        </>
      }
    >
      <XYChart
        series={series}
        segments={basis}
        xRange={[0, 1]}
        yRange={[-2, 2]}
        xLabel="x"
        yLabel="y"
        handles={handles}
      />
    </Interactive>
  )
}
