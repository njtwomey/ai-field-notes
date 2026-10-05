import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  MathText,
  Bars,
  Plot,
  Readout,
  seriesLayers,
  useAxis,
  Slider,
  useFigureState,
  type SeriesSpec,
} from 'aifn-render'
import { useClassColors } from '../_shared/classColor'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf, sigmoid } from 'aifn-compute/numerics/special'

type Link = 'logit' | 'probit'
const LINKS = [
  { value: 'logit' as const, label: 'logit (logistic noise)' },
  { value: 'probit' as const, label: 'probit (Gaussian noise)' },
]
const Z = toFlat(linspace(-7, 7, 281))
const GAP = 0.2
const CLASSES = [1, 2, 3, 4]

const logisticPdf = (z: number) => sigmoid(z) * (1 - sigmoid(z))

/**
 * The latent-variable view of a cumulative link model with four classes: y* = η + ε, and the class is the interval of
 * the real line into which y* falls. The thresholds are draggable; the class probabilities are the areas between them.
 */
export function LatentThresholds() {
  const state = useFigureState({
    eta: float(0.5, { min: -5, max: 5, step: 0.1, label: 'linear predictor η' }),
    link: choice<Link>(LINKS, 'logit', { label: 'link' }),
  })
  const [theta, setTheta] = useState([-2, 0, 1.5])

  // Keep the thresholds ordered, at least GAP apart, and inside the plotted range.
  const setThreshold = (j: number) => (v: number) =>
    setTheta((th) => {
      const lo = j > 0 ? th[j - 1] + GAP : -6
      const hi = j < th.length - 1 ? th[j + 1] - GAP : 6
      const next = th.slice()
      next[j] = Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100
      return next
    })

  const colors = useClassColors(4)
  const [t0, t1, t2] = theta
  const { series, probs } = useMemo(() => {
    const th = [t0, t1, t2]
    const pdf = state.link === 'logit' ? logisticPdf : (v: number) => normalPdf(v)
    const cdf = state.link === 'logit' ? (v: number) => sigmoid(v) : (v: number) => normalCdf(v)
    const edges = [-Infinity, ...th, Infinity]
    // One filled piece of the density per class, so each area reads as that class's probability.
    const pieces: SeriesSpec[] = edges.slice(0, -1).map((lo, k) => {
      const hi = edges[k + 1]
      const zs = [Math.max(lo, -7), ...Z.filter((z) => z > lo && z < hi), Math.min(hi, 7)]
      return {
        name: `class ${k + 1}`,
        type: 'line',
        x: zs,
        y: zs.map((z) => pdf(z - state.eta)),
        area: true,
        color: colors[k],
      }
    })
    const cum = [0, ...th.map((t) => cdf(t - state.eta)), 1]
    return { series: pieces, probs: cum.slice(1).map((c, k) => c - cum[k]) }
  }, [t0, t1, t2, state.eta, state.link, colors])

  const handles: Handle[] = theta.map((t, j) => ({ kind: 'x', at: t, label: `θ${j + 1}`, onDrag: setThreshold(j) }))

  const xAxis = useAxis({ label: 'latent score y*', range: [-7, 7] })
  const yAxis = useAxis({ label: 'density', range: [0, state.link === 'logit' ? 0.27 : 0.42] })
  const kAxis = useAxis({ label: 'class k', range: [0.5, 4.5], integer: true })
  const pAxis = useAxis({ label: 'probability', range: [0, 1] })
  return (
    <Figure
      title="Thresholds cut the latent density into classes"
      state={state}
      caption={
        <MathText text="The latent score $y^* = \eta + \epsilon$ has its noise density centred on the linear predictor $\eta$. The thresholds $\theta_1 < \theta_2 < \theta_3$ cut the real line into four intervals, and each class probability is the area of the density over its interval. Drag the thresholds (or use their sliders); move $\eta$ to shift the whole density. Moving $\eta$ right moves mass towards the higher classes, which is the only way the inputs act." />
      }
      controls={
        <>
          {theta.map((t, j) => (
            <Slider
              key={j}
              label={`threshold θ${j + 1}`}
              value={t}
              onChange={setThreshold(j)}
              min={-6}
              max={6}
              step={0.05}
            />
          ))}
        </>
      }
      readouts={probs.map((p, k) => (
        <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
      ))}
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
        <Plot
          x={xAxis}
          y={yAxis}
          height={280}
          ariaLabel={'Latent noise density split into class regions by thresholds'}
        >
          {seriesLayers(series)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={kAxis} y={pAxis} height={280} ariaLabel="Class probabilities">
          <Bars name="P(y = k)" x={CLASSES} y={probs} colors={colors} />
        </Plot>
      </div>
    </Figure>
  )
}
