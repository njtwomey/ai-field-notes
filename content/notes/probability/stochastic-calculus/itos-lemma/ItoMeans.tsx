import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { joinPaths } from '../_shared/sde'

const PATHS = 2000
const MAX_SHOWN = 50
const STEPS = 200
const T = 2
const TIMES = linspace(0, T, STEPS + 1)

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
  const [which, setWhich] = useState<F>('exp')
  const t = useParam(1, { min: 0, max: T, step: 0.05 })
  const count = useParam(12, { min: 1, max: MAX_SHOWN, step: 1 })

  const paths = useMemo(() => {
    const { normal } = rng(21)
    const sd = Math.sqrt(T / STEPS)
    return Array.from({ length: PATHS }, () => {
      const w = new Float64Array(STEPS + 1)
      for (let k = 1; k <= STEPS; k++) w[k] = w[k - 1] + sd * normal()
      return w
    })
  }, [])

  // The average always uses all 2,000 paths; the paths slider only sets how many are drawn.
  const mean = useMemo(() => {
    const { f } = FUNCS[which]
    return TIMES.map((_, k) => paths.reduce((s, w) => s + f(w[k]), 0) / PATHS)
  }, [which, paths])

  const series = useMemo(() => {
    const { f, ito } = FUNCS[which]
    const out: XYSeries[] = []
    const shown = joinPaths(paths.slice(0, count.value).map((w) => ({ x: TIMES, y: TIMES.map((_, k) => f(w[k])) })))
    out.push({ name: 'paths of f(W_t)', type: 'line', ...shown, slot: 3, thin: count.value > 1 })
    out.push({ name: 'average of 2,000 paths', type: 'line', x: TIMES, y: mean, slot: 0 })
    out.push({ name: "Itô's lemma", type: 'line', x: TIMES, y: TIMES.map(ito), slot: 1, dashed: true })
    out.push({ name: 'ordinary chain rule', type: 'line', x: TIMES, y: TIMES.map(() => f(0)), slot: 2, dashed: true })
    return out
  }, [which, paths, mean, count.value])

  const k = Math.round((t.value / T) * STEPS)
  const { f, ito, label, range } = FUNCS[which]
  return (
    <Interactive
      title="Itô's lemma predicts the mean; the ordinary chain rule does not"
      caption="The light lines are simulated paths of f(Wₜ); the paths slider sets how many are drawn. The solid line averages 2,000 such paths. The ordinary chain rule, df = f′(W) dW, has mean zero and predicts that the average stays at f(0). Itô's lemma adds ½f″(W) dt and predicts t for W², e^{t/2} for eᵂ and e^{−t/2} for cos W; the simulated average follows it. Drag the vertical line to read the values at another time."
      controls={
        <>
          <ParamChoice
            label="f(W)"
            value={which}
            onChange={setWhich}
            options={(Object.keys(FUNCS) as F[]).map((key) => ({ value: key, label: FUNCS[key].label }))}
          />
          <ParamSlider label="time t" param={t} />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label={`average of ${label}`} value={formatNumber(mean[k])} />
          <Readout label="Itô prediction" value={formatNumber(ito(t.value))} />
          <Readout label="chain-rule prediction" value={formatNumber(f(0))} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="t"
        yLabel={`f(W_t) = ${label}`}
        series={series}
        xRange={[0, T]}
        yRange={range}
        handles={[{ kind: 'x', at: t.value, label: 't', onDrag: (x) => t.set(x) }]}
      />
    </Interactive>
  )
}
