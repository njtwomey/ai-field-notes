import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { BOX, Y_RANGE, exact, slackSeries, svmSeries } from './plot'
import { C0, X, Y, position, solve, type Point } from './solver'

const POINTS = X.map((_, i) => ({ value: String(i), label: `x${i + 1}` }))
const snap = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, Math.round(v * 10) / 10))

/** The solved example: boundary, margins and each point's slack, with C on a slider and one point draggable. */
export function SlackExplorer() {
  // log₂ C in steps of 0.1, so that C = 1/2, 1, 2, ... are reachable exactly.
  const state = useFigureState({
    logC: float(Math.log2(C0), {
      min: -4,
      max: 4,
      step: 0.1,
      label: 'C (price of one unit of slack)',
      format: (v) => exact(2 ** v),
    }),
    dragged: choice(POINTS, '3', { label: 'point to drag' }),
  })
  const C = 2 ** state.logC
  const [pts, setPts] = useState<Point[]>(X)

  const r = useMemo(() => {
    const fit = solve(pts, Y, C)
    const margin = pts.map((p, t) => Y[t] * (fit.w[0] * p[0] + fit.w[1] * p[1] + fit.b))
    const xi = margin.map((m) => Math.max(0, 1 - m))
    const norm2 = fit.w[0] ** 2 + fit.w[1] ** 2
    const hinge = xi.reduce((s, v) => s + v, 0)
    return { ...fit, margin, xi, norm2, hinge, primal: norm2 / 2 + C * hinge }
  }, [pts, C])

  const series = useMemo(
    () => [...svmSeries(pts, Y, r.w, r.b), ...slackSeries(pts, Y, r.w, r.b)] as const,
    [pts, r.w, r.b],
  )
  const k = Number(state.dragged)

  const status = (a: number) => (a < 1e-6 ? '0' : a > C - 1e-6 ? 'C' : 'free')

  const xAxis = useAxis({ label: 'x₁', range: BOX.x })
  const yAxis = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="Slack, multipliers and position of each point"
      state={state}
      caption="The solid line is the boundary f(x) = wᵀx + b = 0 and the dashed lines are the margins f = ±1. A coloured segment runs from each point with positive slack to its own margin line; its length in units of f is ξᵢ = max(0, 1 − yᵢf(xᵢ)). The table gives each point's α, y f and ξ. Choose a point and drag it (or press anywhere on the plot) to move it; lower C to widen the margin and pull more points to α = C."
      controls={
        <>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setPts(X)
              state.set('logC', Math.log2(C0))
            }}
          >
            Reset to the example
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="w" value={`(${exact(r.w[0])}, ${exact(r.w[1])})`} />
          <Readout label="b" value={exact(r.b)} />
          <Readout label="margin width 2/‖w‖" value={formatNumber(2 / Math.sqrt(r.norm2))} />
          <Readout label="Σξᵢ" value={exact(r.hinge)} />
          <Readout label="primal ½‖w‖² + CΣξᵢ" value={exact(r.primal)} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="mx-auto w-full max-w-lg">
          <Plot x={xAxis} y={yAxis} ariaLabel={'Six points, the SVM boundary and margins, and slack segments'}>
            {seriesLayers(series)}
            <Handle
              kind="point"
              at={pts[k]}
              label={`x${k + 1}`}
              onDrag={([a, b]) =>
                setPts((prev) => prev.map((p, t) => (t === k ? [snap(a, BOX.x), snap(b, BOX.y)] : p)))
              }
            />
          </Plot>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>point</TableHead>
              <TableHead className="text-right">y</TableHead>
              <TableHead className="text-right">α</TableHead>
              <TableHead className="text-right">y f(x)</TableHead>
              <TableHead className="text-right">ξ</TableHead>
              <TableHead>position</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pts.map((p, t) => (
              <TableRow key={t}>
                <TableCell className="py-1 font-mono text-xs">
                  x{t + 1} ({formatNumber(p[0])}, {formatNumber(p[1])})
                </TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                  {Y[t] > 0 ? '+1' : '−1'}
                </TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                  {exact(r.alpha[t])} <span className="text-muted-foreground">({status(r.alpha[t])})</span>
                </TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{exact(r.margin[t])}</TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{exact(r.xi[t])}</TableCell>
                <TableCell className="py-1 text-xs">{position(r.margin[t])}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
