import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Histogram,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { DATA, experiment, FOLDS, GRID, NAMES, BINS, type DataName } from './widths'

const SLOTS = [2, 3, 4]

/**
 * The median heuristic on an RBF support vector machine: distances over random pairs give three candidate widths
 * (inverse 0.1 quantile, median and 0.9 quantile), each is scored by five-fold cross-validation, and a grid of widths
 * shows where the cross-validated optimum lies.
 */
export function KernelWidthSvm() {
  const state = useFigureState({
    data: choice<DataName>(DATA, 'spirals', { label: 'data set' }),
    n: int(200, { min: 50, max: 400, label: 'points', suggestions: [100, 200, 400] }),
    noise: float(0.1, { min: 0, max: 0.5, step: 0.01, label: 'noise' }),
    C: float(1, { min: 0.01, max: 100, step: 0.01, label: 'SVM penalty C', suggestions: [0.1, 1, 10, 100] }),
    show: choice<'best' | '0' | '1' | '2'>(
      [
        { value: 'best', label: 'best of the three' },
        { value: '0', label: NAMES[0] },
        { value: '1', label: NAMES[1] },
        { value: '2', label: NAMES[2] },
      ],
      'best',
      { label: 'boundary drawn for' },
    ),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const run = useComputed(
    () => experiment({ data: state.data, n: state.n, noise: state.noise, C: state.C, seed: state.seed }),
    [state.data, state.n, state.noise, state.C, state.seed],
    { mode: 'release' },
  )
  const r = run.value
  const shown = state.show === 'best' ? r.best : Number(state.show)
  const field = r.fields[shown]
  const top = Math.max(...field.flat().map(Math.abs))
  const bestGrid = Math.max(...r.grid)

  const xa = useAxis({ label: 'distance between a random pair' })
  const ya = useAxis({ label: 'pairs' })
  // Whole decades on the log axis, so its ticks read 0.01, 0.1, 1 and 10.
  const xb = useAxis({ label: 'kernel lengthscale ℓ', log: true, range: [0.01, 10] })
  const yb = useAxis({ label: `${FOLDS}-fold accuracy`, range: [0.4, 1] })
  const xc = useAxis({ label: 'x₁', range: [r.gx[0], r.gx[r.gx.length - 1]] })
  const yc = useAxis({ label: 'x₂', range: [r.gy[0], r.gy[r.gy.length - 1]], equal: xc })
  return (
    <Figure
      title="Choosing an RBF width for an SVM"
      state={state}
      caption="Left: the decision function of the chosen candidate width, fitted on all the points, with its boundary in white. Right: the five-fold cross-validated accuracy of an RBF support vector machine across kernel widths (grey), with the three candidates the quantiles give (coloured dots). Below: distances between 1000 random pairs of points, with the three quantiles that set the candidates. On the moons, circles and xor the median candidate is at or near the best width on the grid. On the spirals and the checkerboard the classes alternate on a scale far below the typical distance, the median is far too smooth, and only the 0.1-quantile candidate works: the reason to try all three."
      readouts={
        <>
          {r.candidates.map((c, i) => (
            <Readout
              key={NAMES[i]}
              label={`${NAMES[i]}: d = ${formatNumber(r.q[i])}, ℓ = ${formatNumber(c.lengthscale)}`}
              value={`${formatNumber(c.accuracy)} ± ${formatNumber(c.std)}`}
            />
          ))}
          <Readout label="best on the grid" value={formatNumber(bestGrid)} />
          <Readout label="drawn" value={NAMES[shown]} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={xc} y={yc}>
            <Raster
              x={r.gx}
              y={r.gy}
              z={field}
              scale="diverging"
              range={[-top, top]}
              valueLabel="f(x)"
              boundary
              fillOpacity={0.6}
              stale={run.stale}
            />
            <Points
              name="points"
              x={r.X.map((p) => p[0])}
              y={r.X.map((p) => p[1])}
              group={r.y}
              groupNames={['class 0', 'class 1']}
              thin
            />
          </Plot>
          <Plot x={xb} y={yb}>
            <Curve name="accuracy across widths" x={GRID} y={r.grid} muted showPoints stale={run.stale} />
            {r.candidates.map((c, i) => (
              <Points key={NAMES[i]} name={NAMES[i]} x={[c.lengthscale]} y={[c.accuracy]} slot={SLOTS[i]} size={10} />
            ))}
          </Plot>
        </div>
        <Plot x={xa} y={ya} scale={0.4}>
          <Histogram name="pair distances" values={r.dist} bins={BINS} normalize="count" muted stale={run.stale} />
          {r.q.map((v, i) => (
            <Segments
              key={NAMES[i]}
              name={NAMES[i]}
              segments={[{ from: [v, 0], to: [v, r.histTop] }]}
              slot={SLOTS[i]}
              width={2}
            />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
