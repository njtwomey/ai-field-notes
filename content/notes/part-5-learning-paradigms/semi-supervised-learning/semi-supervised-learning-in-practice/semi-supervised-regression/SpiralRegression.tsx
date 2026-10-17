import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { fromRows } from 'aifn-compute/foundation/tensor'
import { fitSpiral, POINTS, spiralData, spiralLaplacian, spreadLabels } from './spiral-study'

const RANGE: [number, number] = [-3.5, 3.5]
const NG = 36
const G = Array.from({ length: NG }, (_, j) => RANGE[0] + ((j + 0.5) * (RANGE[1] - RANGE[0])) / NG)
const GRID = fromRows(G.flatMap((gy) => G.map((gx) => [gx, gy])))
const SWEEP = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]

/**
 * Semi-supervised regression on a spiral: kernel ridge regression on the labelled points against Laplacian
 * regularised least squares on a graph of all points, as the intrinsic weight grows.
 */
export function SpiralRegression() {
  const fig = useFigureState({
    gammaI: float(1e4, { gt: 0, le: 1e7, scale: 'log10', label: 'intrinsic weight γ_I', onChart: true }),
    gammaA: float(1e-3, {
      gt: 0,
      le: 10,
      scale: 'log10',
      label: 'ambient weight γ_A',
      suggestions: [1e-4, 1e-3, 1e-2],
    }),
    lengthscale: float(0.6, { gt: 0, le: 5, step: 0.05, label: 'kernel lengthscale ℓ', suggestions: [0.3, 0.6, 1.2] }),
    k: int(6, { min: 2, max: 30, label: 'graph neighbours k', suggestions: [4, 6, 10] }),
    labels: int(5, { min: 2, max: 40, label: 'labelled points', suggestions: [3, 5, 10, 20] }),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { gammaI, gammaA, lengthscale, k, labels, seed } = fig
  const data = useMemo(() => spiralData(seed), [seed])
  const laplacian = useMemo(() => spiralLaplacian(data.x, k), [data, k])
  const labelled = useMemo(() => spreadLabels(labels), [labels])

  const fits = useComputed(
    () => ({
      ridge: fitSpiral(data, laplacian, labelled, { lengthscale, gammaA, gammaI: 0 }),
      lap: fitSpiral(data, laplacian, labelled, { lengthscale, gammaA, gammaI, extra: GRID }),
    }),
    [data, laplacian, labelled, lengthscale, gammaA, gammaI],
  )
  const sweep = useComputed(
    () => SWEEP.map((g) => fitSpiral(data, laplacian, labelled, { lengthscale, gammaA, gammaI: g }).rmse),
    [data, laplacian, labelled, lengthscale, gammaA],
    { mode: 'release' },
  )
  const { ridge, lap } = fits.value

  const xa = useAxis({ label: 'x₁', range: RANGE })
  const ya = useAxis({ label: 'x₂', range: RANGE, equal: xa })
  const xp = useAxis({ label: 'true position along the spiral', range: [-1.05, 1.05] })
  const yp = useAxis({ label: 'prediction', range: [-1.6, 1.6] })
  const xg = useAxis({ label: 'intrinsic weight γ_I', log: true, range: [1, 1e6] })
  const yg = useAxis({ label: 'RMSE on unlabelled points', range: [0, 0.6] })
  const field = G.map((_, i) => lap.extra.slice(i * NG, (i + 1) * NG))

  return (
    <Figure
      title="Smooth along the curve, not across it"
      state={fig}
      caption={`${POINTS} points along a spiral; the target of each is its position along the curve, from −1 at the inner end to 1 at the outer end. A few points spread along the curve are labelled (ink). Left: Laplacian regularised least squares (LapRLS) on a k-nearest-neighbour graph of all points, shaded over the plane, with the points. Middle: predictions at every point against the true position, for kernel ridge regression on the labelled points alone (γ_I = 0) and for LapRLS; the diagonal is a perfect fit. Right: the error on the unlabelled points as the intrinsic weight grows; drag the vertical line to set γ_I. Kernel ridge regression measures distance in the plane, so it shrinks to zero between labels and lets neighbouring arms of the spiral share values. The graph penalty measures change along the curve, so LapRLS interpolates between labels in arc length. Too large a γ_I flattens the fit towards one constant, which fits the labels badly.`}
      readouts={
        <>
          <Readout label="RMSE, kernel ridge" value={formatNumber(ridge.rmse)} />
          <Readout label="RMSE, LapRLS" value={formatNumber(lap.rmse)} />
          <Readout label="γ_I / n²" value={formatNumber(gammaI / POINTS ** 2)} />
        </>
      }
    >
      <Plots cols={3}>
        <Plot x={xa} y={ya} title="LapRLS over the plane">
          <Raster
            x={G}
            y={G}
            z={field}
            scale="diverging"
            range={[-1, 1]}
            valueLabel="f(x)"
            fillOpacity={0.6}
            stale={fits.stale}
          />
          <Points name="points" x={data.xy.map((p) => p[0])} y={data.xy.map((p) => p[1])} muted thin />
          <Points
            name="labelled"
            x={labelled.map((i) => data.xy[i][0])}
            y={labelled.map((i) => data.xy[i][1])}
            emphasis
            size={10}
          />
        </Plot>
        <Plot x={xp} y={yp} title="predictions along the curve">
          <Curve name="perfect fit" x={[-1, 1]} y={[-1, 1]} muted dashed />
          <Curve name="kernel ridge (γ_I = 0)" x={data.y} y={ridge.fitted} slot={0} stale={fits.stale} />
          <Curve name="LapRLS" x={data.y} y={lap.fitted} slot={1} stale={fits.stale} />
          <Points
            name="labelled"
            x={labelled.map((i) => data.y[i])}
            y={labelled.map((i) => data.y[i])}
            emphasis
            size={10}
          />
        </Plot>
        <Plot x={xg} y={yg} title="error against γ_I">
          <Curve name="LapRLS" x={SWEEP} y={sweep.value} slot={1} showPoints stale={sweep.stale} />
          <Curve name="kernel ridge" x={[1, 1e6]} y={[ridge.rmse, ridge.rmse]} slot={0} dashed />
          <Handle
            kind="x"
            at={Math.min(1e6, Math.max(1, gammaI))}
            label="γ_I"
            onDrag={(v) => fig.set('gammaI', Number(Math.min(1e6, Math.max(1, v)).toPrecision(2)))}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
