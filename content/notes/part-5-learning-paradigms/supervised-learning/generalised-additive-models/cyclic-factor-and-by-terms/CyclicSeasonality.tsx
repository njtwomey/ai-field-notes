import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  addScaled,
  cholesky,
  cholSolve,
  crossprod,
  crossprodY,
  cyclicSecondDiff,
  diffMatrix,
  dot,
  gram,
  periodicRow,
  psplineRow,
  traceSolve,
} from '../_shared/terms-psplines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 200
const YEAR = 365
const NOISE = 1.5
// A seasonal cycle whose low point falls near the turn of the year, where the two ends of an ordinary spline meet.
const truth = (d: number) =>
  8 + 6 * Math.cos((2 * Math.PI * (d - 200)) / YEAR) + 1.5 * Math.sin((4 * Math.PI * d) / YEAR)
const GRID = toFlat(linspace(0, 2 * YEAR, 293))

export function CyclicSeasonality() {
  const state = useFigureState({
    logLambda: float(0, { min: -2, max: 5, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    k: int(12, { min: 6, max: 24, step: 1, label: 'basis size k' }),
    seed: int(7, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const day = Array.from({ length: N }, () => uniform(r) * YEAR)
    return { day, y: day.map((d) => truth(d) + NOISE * normal(r)) }
  }, [state.seed])

  const bases = useMemo(() => {
    const ordinary = data.day.map((d) => psplineRow(d, 0, YEAR, state.k))
    const cyclic = data.day.map((d) => periodicRow(d, 0, YEAR, state.k))
    return {
      ordinary: { G: crossprod(ordinary), b: crossprodY(ordinary, data.y), S: gram(diffMatrix(state.k, 2)) },
      cyclic: { G: crossprod(cyclic), b: crossprodY(cyclic, data.y), S: gram(cyclicSecondDiff(state.k)) },
    }
  }, [data, state.k])

  const lambda = 10 ** state.logLambda
  const fits = useMemo(() => {
    const fit = ({ G, b, S }: { G: number[][]; b: number[]; S: number[][] }) => {
      const L = cholesky(addScaled(G, [lambda, S]))
      return { beta: cholSolve(L, b), edf: traceSolve(L, G) }
    }
    return { ordinary: fit(bases.ordinary), cyclic: fit(bases.cyclic) }
  }, [bases, lambda])

  // Both fits are defined on one year; the second year repeats the first, as a model of day-of-year does.
  const ordinaryAt = (d: number) => dot(psplineRow(d % YEAR, 0, YEAR, state.k), fits.ordinary.beta)
  const cyclicAt = (d: number) => dot(periodicRow(d % YEAR, 0, YEAR, state.k), fits.cyclic.beta)
  const ordinaryEnd = dot(psplineRow(YEAR, 0, YEAR, state.k), fits.ordinary.beta)
  const jump = ordinaryAt(0) - ordinaryEnd
  const eps = 0.01
  const slopeJump =
    (ordinaryAt(eps) - ordinaryAt(0)) / eps -
    (ordinaryEnd - dot(psplineRow(YEAR - eps, 0, YEAR, state.k), fits.ordinary.beta)) / eps

  const series = [
    {
      name: 'data',
      x: [...data.day, ...data.day.map((d) => d + YEAR)],
      y: [...data.y, ...data.y],
      muted: true,
    },
    { name: 'ordinary spline', x: GRID, y: GRID.map(ordinaryAt), slot: 0 },
    { name: 'cyclic spline', x: GRID, y: GRID.map(cyclicAt), slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'day (two years)', range: [0, 2 * YEAR] })
  const yAxis = useAxis({ label: 'y', range: [-2, 18] })
  return (
    <Figure
      title="Day-of-year seasonality: cyclic versus ordinary spline"
      state={state}
      caption={
        <>
          Two hundred noisy observations of a seasonal cycle, plotted for two consecutive years because day of year
          wraps from 365 back to 0. Both fits use k evenly spaced cubic B-splines and a second-difference penalty. The
          ordinary spline treats 31 December and 1 January as the two far ends of its range, so it jumps where the years
          meet. The cyclic spline wraps its basis and its penalty around the circle, so its value, slope and curvature
          match at the join. As λ grows the cyclic fit flattens to the mean, while the ordinary fit tends to a straight
          line with a jump of its full rise at the boundary.
        </>
      }

      readouts={
        <>
          <Readout label="ordinary: jump f(0) − f(365)" value={formatNumber(jump)} />
          <Readout label="ordinary: slope jump per day" value={formatNumber(slopeJump)} />
          <Readout label="cyclic: jump" value="0" />
          <Readout
            label="edf ordinary, cyclic"
            value={`${formatNumber(fits.ordinary.edf)}, ${formatNumber(fits.cyclic.edf)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}
