import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
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

const N = 200
const YEAR = 365
const NOISE = 1.5
// A seasonal cycle whose low point falls near the turn of the year, where the two ends of an ordinary spline meet.
const truth = (d: number) =>
  8 + 6 * Math.cos((2 * Math.PI * (d - 200)) / YEAR) + 1.5 * Math.sin((4 * Math.PI * d) / YEAR)
const GRID = linspace(0, 2 * YEAR, 293)

export function CyclicSeasonality() {
  const [k, setK] = useState(12)
  const [logLambda, setLogLambda] = useState(0)
  const [seed, setSeed] = useState(7)

  const data = useMemo(() => {
    const r = rng(seed)
    const day = Array.from({ length: N }, () => r.uniform() * YEAR)
    return { day, y: day.map((d) => truth(d) + NOISE * r.normal()) }
  }, [seed])

  const bases = useMemo(() => {
    const ordinary = data.day.map((d) => psplineRow(d, 0, YEAR, k))
    const cyclic = data.day.map((d) => periodicRow(d, 0, YEAR, k))
    return {
      ordinary: { G: crossprod(ordinary), b: crossprodY(ordinary, data.y), S: gram(diffMatrix(k, 2)) },
      cyclic: { G: crossprod(cyclic), b: crossprodY(cyclic, data.y), S: gram(cyclicSecondDiff(k)) },
    }
  }, [data, k])

  const lambda = 10 ** logLambda
  const fits = useMemo(() => {
    const fit = ({ G, b, S }: { G: number[][]; b: number[]; S: number[][] }) => {
      const L = cholesky(addScaled(G, [lambda, S]))
      return { beta: cholSolve(L, b), edf: traceSolve(L, G) }
    }
    return { ordinary: fit(bases.ordinary), cyclic: fit(bases.cyclic) }
  }, [bases, lambda])

  // Both fits are defined on one year; the second year repeats the first, as a model of day-of-year does.
  const ordinaryAt = (d: number) => dot(psplineRow(d % YEAR, 0, YEAR, k), fits.ordinary.beta)
  const cyclicAt = (d: number) => dot(periodicRow(d % YEAR, 0, YEAR, k), fits.cyclic.beta)
  const ordinaryEnd = dot(psplineRow(YEAR, 0, YEAR, k), fits.ordinary.beta)
  const jump = ordinaryAt(0) - ordinaryEnd
  const eps = 0.01
  const slopeJump =
    (ordinaryAt(eps) - ordinaryAt(0)) / eps -
    (ordinaryEnd - dot(psplineRow(YEAR - eps, 0, YEAR, k), fits.ordinary.beta)) / eps

  const series: XYSeries[] = [
    {
      name: 'data',
      type: 'scatter',
      x: [...data.day, ...data.day.map((d) => d + YEAR)],
      y: [...data.y, ...data.y],
      muted: true,
    },
    { name: 'ordinary spline', type: 'line', x: GRID, y: GRID.map(ordinaryAt), slot: 0 },
    { name: 'cyclic spline', type: 'line', x: GRID, y: GRID.map(cyclicAt), slot: 1 },
  ]

  return (
    <Interactive
      title="Day-of-year seasonality: cyclic versus ordinary spline"
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
      controls={
        <>
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-2}
            max={5}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider label="basis size k" value={k} onChange={setK} min={6} max={24} step={1} withArrows />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
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
      <XYChart
        series={series}
        xRange={[0, 2 * YEAR]}
        yRange={[-2, 18]}
        xLabel="day (two years)"
        yLabel="y"
        height={320}
      />
    </Interactive>
  )
}
