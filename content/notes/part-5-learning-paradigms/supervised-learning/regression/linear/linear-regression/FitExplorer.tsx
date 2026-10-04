import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { lstsq } from 'aifn/numerics/linalg'
import { linspace, tensor, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const TRUE_SLOPE = 1.5
const TRUE_INTERCEPT = -0.5
/** x position of the slope handle. The intercept handle sits at x = 0. */
const SLOPE_AT = 2

/** The least-squares line through (xᵢ, yᵢ): aifn's lstsq on the design matrix [xᵢ, 1]. */
function olsFit(x: number[], y: number[]) {
  const [slope, intercept] = toFlat(lstsq(tensor(x.map((xi) => [xi, 1])), tensor(y)).x)
  return { slope, intercept }
}

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length

export function FitExplorer() {
  const state = useFigureState({
    slopeParam: float(0.5, { min: -2, max: 3, step: 0.05, label: 'slope' }),
    interceptParam: float(0.5, { min: -3, max: 3, step: 0.05, label: 'intercept' }),
    noise: float(0.8, { min: 0, max: 3, step: 0.1, label: 'noise σ' }),
    n: int(40, { min: 5, max: 200, step: 1, label: 'samples' }),
    showOls: setting(true, 'show least-squares fit'),
  })
  const slope = state.slopeParam
  const intercept = state.interceptParam

  const data = useMemo(() => {
    const r = stream(7)
    const x = Array.from({ length: state.n }, () => -3 + 6 * uniform(r))
    const eps = Array.from({ length: state.n }, () => normal(r))
    return { x, y: x.map((xi, i) => TRUE_SLOPE * xi + TRUE_INTERCEPT + state.noise * eps[i]) }
  }, [state.n, state.noise])

  const ols = useMemo(() => olsFit(data.x, data.y), [data])
  const mse = (m: number, c: number) => mean(data.x.map((xi, i) => (data.y[i] - (m * xi + c)) ** 2))
  const grid = toFlat(linspace(-3, 3, 2))
  // Two points on the line: (0, b) moves the intercept, (2, 2m + b) turns the line about the intercept.

  const xAxis = useAxis({ label: 'x', range: [-3, 3] })
  const yAxis = useAxis({ label: 'y', range: [-9, 6] })
  return (
    <Figure
      title="Fit a line by hand"
      state={state}
      caption="Move the slope and intercept with the sliders, or drag the two dots on the line: the dot at x = 0 sets the intercept, the other turns the line. The grey segments are residuals. Least squares minimises the mean of their squared lengths."

      readouts={
        <>
          <Readout label="your MSE" value={formatNumber(mse(slope, intercept))} />
          <Readout label="least-squares MSE" value={formatNumber(mse(ols.slope, ols.intercept))} />
          <Readout
            label="least-squares fit"
            value={`y = ${formatNumber(ols.slope)}x ${ols.intercept < 0 ? '−' : '+'} ${formatNumber(Math.abs(ols.intercept))}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers([
          { name: 'data', type: 'scatter', x: data.x, y: data.y, slot: 0 },
          { name: 'your line', type: 'line', x: grid, y: grid.map((g) => slope * g + intercept), slot: 1 },
          ...(state.showOls
            ? [
                {
                  name: 'least squares',
                  type: 'line' as const,
                  x: grid,
                  y: grid.map((g) => ols.slope * g + ols.intercept),
                  slot: 2,
                  dashed: true,
                },
              ]
            : []),
        ])}
        <Segments segments={data.x.map((xi, i) => ({ from: [xi, data.y[i]], to: [xi, slope * xi + intercept] }))} />
        <Handle kind="point" at={[0, intercept]} label="intercept" onDrag={([, y]) => state.set('interceptParam', y)} />
        <Handle
          kind="point"
          at={[SLOPE_AT, SLOPE_AT * slope + intercept]}
          label="slope"
          onDrag={([, y]) => state.set('slopeParam', (y - intercept) / SLOPE_AT)}
        />
      </Plot>
    </Figure>
  )
}
