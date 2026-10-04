import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { joinPaths } from '../_shared/sde'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'

const PATHS = 2000
const MAX_SHOWN = 50
const STEPS = 200
const T = 2
const TIMES = toFlat(linspace(0, T, STEPS + 1))

type F = 'square' | 'exp' | 'cos'
const FUNCS: Record<
  F,
  { f: (w: number) => number; ito: (t: number) => number; label: string; range: [number, number] }
> = {
  square: { f: (w) => w * w, ito: (t) => t, label: 'W²', range: [0, 6] },
  exp: { f: (w) => Math.exp(w), ito: (t) => Math.exp(t / 2), label: 'eᵂ', range: [0, 6] },
  cos: { f: (w) => Math.cos(w), ito: (t) => Math.exp(-t / 2), label: 'cos W', range: [-1, 1] },
}

/**
 * Simulated paths of f(W_t) and their average over 2,000 paths, against two predictions of the mean: the ordinary
 * chain rule (the mean stays at f(0)) and Itô's lemma (the mean moves by ½ E f''(W) dt).
 */
export function ItoMeans() {
  const state = useFigureState({
    which: choice<F>(
      (Object.keys(FUNCS) as F[]).map((key) => ({ value: key, label: FUNCS[key].label })),
      'exp',
      { label: 'f(W)' },
    ),
    t: slider(0, T, 1, { step: 0.05, label: 'time t' }),
    count: int(12, { min: 1, max: MAX_SHOWN, step: 1, label: 'paths', format: (v) => String(v) }),
  })

  const paths = useMemo(() => {
    const rs = stream(21)
    const sd = Math.sqrt(T / STEPS)
    return Array.from({ length: PATHS }, () => {
      const w = new Float64Array(STEPS + 1)
      for (let k = 1; k <= STEPS; k++) w[k] = w[k - 1] + sd * normal(rs)
      return w
    })
  }, [])

  // The average always uses all 2,000 paths; the paths slider only sets how many are drawn.
  const mean = useMemo(() => {
    const { f } = FUNCS[state.which]
    return TIMES.map((_, k) => paths.reduce((s, w) => s + f(w[k]), 0) / PATHS)
  }, [state.which, paths])

  const series = useMemo(() => {
    const { f, ito } = FUNCS[state.which]
    const out: SeriesSpec[] = []
    const shown = joinPaths(paths.slice(0, state.count).map((w) => ({ x: TIMES, y: TIMES.map((_, k) => f(w[k])) })))
    out.push({ name: 'paths of f(W_t)', type: 'line', ...shown, slot: 3, thin: state.count > 1 })
    out.push({ name: 'average of 2,000 paths', type: 'line', x: TIMES, y: mean, slot: 0 })
    out.push({ name: "Itô's lemma", type: 'line', x: TIMES, y: TIMES.map(ito), slot: 1, dashed: true })
    out.push({ name: 'ordinary chain rule', type: 'line', x: TIMES, y: TIMES.map(() => f(0)), slot: 2, dashed: true })
    return out
  }, [state.which, paths, mean, state.count])

  const k = Math.round((state.t / T) * STEPS)
  const { f, ito, label, range } = FUNCS[state.which]
  const xAxis = useAxis({ label: 't', range: [0, T] })
  const yAxis = useAxis({ label: `f(W_t) = ${label}`, range: range })
  return (
    <Figure
      title="Itô's lemma predicts the mean; the ordinary chain rule does not"
      state={state}
      caption="The light lines are simulated paths of f(Wₜ); the paths slider sets how many are drawn. The solid line averages 2,000 such paths. The ordinary chain rule, df = f′(W) dW, has mean zero and predicts that the average stays at f(0). Itô's lemma adds ½f″(W) dt and predicts t for W², e^{t/2} for eᵂ and e^{−t/2} for cos W; the simulated average follows it. Drag the vertical line to read the values at another time."

      readouts={
        <>
          <Readout label={`average of ${label}`} value={formatNumber(mean[k])} />
          <Readout label="Itô prediction" value={formatNumber(ito(state.t))} />
          <Readout label="chain-rule prediction" value={formatNumber(f(0))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
        <Handle {...state.handle('t', { label: 't' })} />
      </Plot>
    </Figure>
  )
}
