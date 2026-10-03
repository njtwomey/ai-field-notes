import { useMemo, useState } from 'react'
import { MathText } from 'aifn-render'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace, sigmoid } from '@/lib/math'
import { useClassColors } from '../_shared/classColor'
import { adjacentProbs, continuationProbs, cumulativeProbs } from '../_shared/ordinal'

type Family = 'cumulative' | 'continuation' | 'adjacent'
const FAMILIES = [
  { value: 'cumulative' as const, label: 'cumulative' },
  { value: 'continuation' as const, label: 'continuation ratio' },
  { value: 'adjacent' as const, label: 'adjacent category' },
]
const ETA = linspace(-6, 6, 241)
const THETA = [-2, 0, 1.5]

const probsOf = (family: Family, eta: number) =>
  family === 'cumulative'
    ? cumulativeProbs(sigmoid, eta, THETA)
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
  const [family, setFamily] = useState<Family>('continuation')
  const eta = useParam(0, { min: -6, max: 6, step: 0.05 })
  const colors = useClassColors(4)

  const series = useMemo<XYSeries[]>(() => {
    const table = ETA.map((e) => probsOf(family, e))
    return [0, 1, 2, 3].map((k) => ({
      name: `P(y = ${k + 1})`,
      type: 'line',
      x: ETA,
      y: table.map((p) => p[k]),
      color: colors[k],
    }))
  }, [family, colors])

  const handles: Handle[] = [{ kind: 'x', at: eta.value, label: 'η', onDrag: eta.set }]
  const here = logits(family, eta.value)

  return (
    <Interactive
      title="Three ways to split an ordered outcome into binary comparisons"
      caption={
        <MathText text="Four classes, thresholds $\theta = (-2, 0, 1.5)$ and one linear predictor $\eta$. The cumulative family compares $y \le k$ with $y > k$; the continuation ratio compares $y = k$ with $y > k$ among outcomes that reached $k$; the adjacent-category family compares $y = k + 1$ with $y = k$. The readouts show each family's three binary logits at the cursor: each is $\pm(\eta - \theta_k)$ by construction. Drag the cursor along $\eta$." />
      }
      controls={
        <>
          <ParamChoice label="family" value={family} onChange={setFamily} options={FAMILIES} />
          <ParamSlider label="linear predictor η" param={eta} />
        </>
      }
      readout={here.map((l) => (
        <Readout key={l.label} label={l.label} value={formatNumber(l.value)} />
      ))}
    >
      <XYChart
        series={series}
        handles={handles}
        xRange={[-6, 6]}
        yRange={[0, 1]}
        xLabel="linear predictor η"
        yLabel="class probability"
        height={300}
        ariaLabel="Class probability curves for the chosen ordinal family"
      />
    </Interactive>
  )
}
