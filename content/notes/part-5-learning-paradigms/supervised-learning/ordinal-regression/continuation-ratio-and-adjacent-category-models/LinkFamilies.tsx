import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { useClassColors } from '../_shared/classColor'
import { adjacentProbs, continuationProbs, cumulativeProbs } from '../_shared/ordinal'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { sigmoid } from 'aifn/numerics/special'

type Family = 'cumulative' | 'continuation' | 'adjacent'
const FAMILIES = [
  { value: 'cumulative' as const, label: 'cumulative' },
  { value: 'continuation' as const, label: 'continuation ratio' },
  { value: 'adjacent' as const, label: 'adjacent category' },
]
const ETA = toFlat(linspace(-6, 6, 241))
const THETA = [-2, 0, 1.5]

const probsOf = (family: Family, eta: number) =>
  family === 'cumulative'
    ? cumulativeProbs((v: number) => sigmoid(v), eta, THETA)
    : family === 'continuation'
      ? continuationProbs(eta, THETA)
      : adjacentProbs(eta, THETA)

/** The binary logits each family sets equal to θ_k − η (or η − θ_k), evaluated at one η. */
function logits(family: Family, eta: number): { label: string; value: number }[] {
  const p = probsOf(family, eta)
  const logit = (q: number) => Math.log(q / (1 - q))
  return THETA.map((_, k) => {
    const upTo = p.slice(0, k + 1).reduce((a, b) => a + b, 0)
    if (family === 'cumulative') return { label: `logit P(y ≤ ${k + 1})`, value: logit(upTo) }
    if (family === 'continuation') {
      const atLeast = p.slice(k).reduce((a, b) => a + b, 0)
      return { label: `logit P(y = ${k + 1} | y ≥ ${k + 1})`, value: logit(p[k] / atLeast) }
    }
    return { label: `log P(y = ${k + 2})/P(y = ${k + 1})`, value: Math.log(p[k + 1] / p[k]) }
  })
}

/**
 * Class probabilities against the linear predictor for three ordinal families that share the same thresholds and
 * predictor. Each family is logistic regression on a different set of binary comparisons.
 */
export function LinkFamilies() {
  const state = useFigureState({
    family: choice<Family>(FAMILIES, 'continuation', { label: 'family' }),
    eta: float(0, { min: -6, max: 6, step: 0.05, label: 'linear predictor η' }),
  })
  const colors = useClassColors(4)

  const series = useMemo<SeriesSpec[]>(() => {
    const table = ETA.map((e) => probsOf(state.family, e))
    return [0, 1, 2, 3].map((k) => ({
      name: `P(y = ${k + 1})`,
      type: 'line',
      x: ETA,
      y: table.map((p) => p[k]),
      color: colors[k],
    }))
  }, [state.family, colors])

  const here = logits(state.family, state.eta)

  const xAxis = useAxis({ label: 'linear predictor η', range: [-6, 6] })
  const yAxis = useAxis({ label: 'class probability', range: [0, 1] })
  return (
    <Figure
      title="Three ways to split an ordered outcome into binary comparisons"
      state={state}
      caption={
        <MathText text="Four classes, thresholds $\theta = (-2, 0, 1.5)$ and one linear predictor $\eta$. The cumulative family compares $y \le k$ with $y > k$; the continuation ratio compares $y = k$ with $y > k$ among outcomes that reached $k$; the adjacent-category family compares $y = k + 1$ with $y = k$. The readouts show each family's three binary logits at the cursor: each is $\pm(\eta - \theta_k)$ by construction. Drag the cursor along $\eta$." />
      }

      readouts={here.map((l) => (
        <Readout key={l.label} label={l.label} value={formatNumber(l.value)} />
      ))}
    >
      <Plot x={xAxis} y={yAxis} height={300} ariaLabel={'Class probability curves for the chosen ordinal family'}>
        {seriesLayers(series)}
        <Handle {...state.handle('eta', { label: 'η' })} />
      </Plot>
    </Figure>
  )
}
