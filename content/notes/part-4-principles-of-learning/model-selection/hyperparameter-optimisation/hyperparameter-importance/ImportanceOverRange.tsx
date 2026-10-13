import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  Segments,
  slider,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { quantile } from 'aifn-compute/probability/stats'
import { svmTuningSurface } from 'aifn-methods/data/objectives'
import { fanova } from 'aifn-methods/optimisation/importance'

/** The tabulated surface: rows are log10 C, columns log10 gamma, values the cross-validated error. */
const SURFACE = svmTuningSurface()
const LOG_C = Array.from(toFlat(SURFACE.logC))
const LOG_GAMMA = Array.from(toFlat(SURFACE.logGamma))
const ERROR = toArray(SURFACE.error) as number[][]
const STEP = 0.25
const HALF = STEP / 2

/** The grid indices from lo to hi on one axis, at least two of them. */
function span(values: number[], lo: number, hi: number) {
  const index = (v: number) =>
    values.reduce((best, x, i) => (Math.abs(x - v) < Math.abs(values[best] - v) ? i : best), 0)
  let [a, b] = [index(Math.min(lo, hi)), index(Math.max(lo, hi))]
  if (a === b) [a, b] = b < values.length - 1 ? [a, b + 1] : [a - 1, b]
  return Array.from({ length: b - a + 1 }, (_, k) => a + k)
}

/** Functional ANOVA of the SVM error surface inside a draggable box of the hyperparameter grid. */
export function ImportanceOverRange() {
  const state = useFigureState({
    gLo: slider(-3, 2, -3, { step: STEP, onChart: true, label: 'lowest log₁₀ γ' }),
    cLo: slider(-2, 3, -2, { step: STEP, onChart: true, label: 'lowest log₁₀ C' }),
    gHi: slider(-3, 2, 2, { step: STEP, onChart: true, label: 'highest log₁₀ γ' }),
    cHi: slider(-2, 3, 3, { step: STEP, onChart: true, label: 'highest log₁₀ C' }),
    cap: toggle(false, { label: 'cap at the 25% quantile' }),
  })
  const { gLo, gHi, cLo, cHi, cap } = state

  const result = useMemo(() => {
    const rows = span(LOG_C, cLo, cHi)
    const cols = span(LOG_GAMMA, gLo, gHi)
    const raw = rows.map((i) => cols.map((j) => ERROR[i][j]))
    const ceiling = cap ? quantile(raw.flat(), 0.25) : Infinity
    const table = raw.map((row) => row.map((v) => Math.min(v, ceiling)))
    const r = fanova(table, { axes: ['log₁₀ C', 'log₁₀ γ'] })
    return {
      cs: rows.map((i) => LOG_C[i]),
      gs: cols.map((j) => LOG_GAMMA[j]),
      r,
      ceiling,
      best: Math.min(...raw.flat()),
    }
  }, [gLo, gHi, cLo, cHi, cap])

  const { cs, gs, r } = result
  const [x0, x1] = [gs[0] - HALF, gs[gs.length - 1] + HALF]
  const [y0, y1] = [cs[0] - HALF, cs[cs.length - 1] + HALF]
  const box = [
    { from: [x0, y0], to: [x1, y0] },
    { from: [x1, y0], to: [x1, y1] },
    { from: [x1, y1], to: [x0, y1] },
    { from: [x0, y1], to: [x0, y0] },
  ] as const

  const gx = useAxis({ label: 'log₁₀ γ', range: [-3.125, 2.125] })
  const gy = useAxis({ label: 'log₁₀ C', range: [-2.125, 3.125], equal: gx })
  const mx = useAxis({ label: 'log₁₀ value of the hyperparameter', range: [-3.125, 3.125] })
  const my = useAxis({ label: 'mean validation error', range: [0, 0.5] })
  const percent = (v: number) => `${(100 * v).toFixed(1)}%`
  return (
    <Figure
      title="Hyperparameter importance depends on the range"
      state={state}
      caption="Left: the cross-validated error of an RBF support vector machine on two iris classes, tabulated on a 21 × 21 grid of log₁₀ C and log₁₀ γ. Drag the two corners of the box to choose the range over which importance is measured. Right: the marginal error of each hyperparameter, averaged over the other inside the box. The readouts give the fraction of the variance of the error inside the box explained by each main effect and by the interaction, from the exact functional ANOVA of the table. Over the whole grid the cliff at large γ dominates and γ explains 95%. Drag the upper γ corner to 0 to remove the cliff, or cap the error at its 25% quantile to measure importance among good settings only: the diagonal valley then makes the interaction the largest term."
      readouts={{
        importance: (
          <>
            <Readout label="log₁₀ C" value={percent(r.main[0].fraction)} />
            <Readout label="log₁₀ γ" value={percent(r.main[1].fraction)} />
            <Readout label="interaction" value={percent(r.pairs[0].fraction)} />
          </>
        ),
        box: (
          <>
            <Readout label="grid points" value={`${cs.length} × ${gs.length}`} />
            <Readout label="mean error" value={formatNumber(r.mean)} />
            <Readout label="best error" value={formatNumber(result.best)} />
            {cap && <Readout label="cap" value={formatNumber(result.ceiling)} />}
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={gx} y={gy} title="validation error over the grid">
          <Raster x={LOG_GAMMA} y={LOG_C} z={ERROR} valueLabel="validation error" />
          <Segments segments={box} width={2} emphasis />
          <Handle {...state.handle(['gLo', 'cLo'], { label: 'corner' })} />
          <Handle {...state.handle(['gHi', 'cHi'], { label: 'corner' })} />
        </Plot>
        <Plot x={mx} y={my} title="marginal error inside the box">
          <Curve name="log₁₀ C" x={cs} y={Array.from(toFlat(r.main[0].marginal))} slot={0} showPoints />
          <Curve name="log₁₀ γ" x={gs} y={Array.from(toFlat(r.main[1].marginal))} slot={1} showPoints />
        </Plot>
      </div>
    </Figure>
  )
}
