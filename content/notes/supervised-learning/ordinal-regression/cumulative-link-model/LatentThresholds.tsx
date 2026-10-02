import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace, sigmoid } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'
import { useClassColors } from '../_shared/classColor'

type Link = 'logit' | 'probit'
const LINKS = [
  { value: 'logit' as const, label: 'logit (logistic noise)' },
  { value: 'probit' as const, label: 'probit (Gaussian noise)' },
]
const Z = linspace(-7, 7, 281)
const GAP = 0.2

const logisticPdf = (z: number) => sigmoid(z) * (1 - sigmoid(z))

/**
 * The latent-variable view of a cumulative link model with four classes: y* = η + ε, and the class is the interval of
 * the real line into which y* falls. The thresholds are draggable; the class probabilities are the areas between them.
 */
export function LatentThresholds() {
  const [eta, setEta] = useState(0.5)
  const [link, setLink] = useState<Link>('logit')
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
    const pdf = link === 'logit' ? logisticPdf : normalPdf
    const cdf = link === 'logit' ? sigmoid : normalCdf
    const edges = [-Infinity, ...th, Infinity]
    // One filled piece of the density per class, so each area reads as that class's probability.
    const pieces: XYSeries[] = edges.slice(0, -1).map((lo, k) => {
      const hi = edges[k + 1]
      const zs = [Math.max(lo, -7), ...Z.filter((z) => z > lo && z < hi), Math.min(hi, 7)]
      return {
        name: `class ${k + 1}`,
        type: 'line',
        x: zs,
        y: zs.map((z) => pdf(z - eta)),
        area: true,
        color: colors[k],
      }
    })
    const cum = [0, ...th.map((t) => cdf(t - eta)), 1]
    return { series: pieces, probs: cum.slice(1).map((c, k) => c - cum[k]) }
  }, [t0, t1, t2, eta, link, colors])

  const handles: Handle[] = theta.map((t, j) => ({ kind: 'x', at: t, label: `θ${j + 1}`, onDrag: setThreshold(j) }))
  const bars: XYSeries[] = [{ name: 'P(y = k)', type: 'bar', x: [1, 2, 3, 4], y: probs, pointColors: colors }]

  return (
    <Interactive
      title="Thresholds cut the latent density into classes"
      caption={
        <MathText text="The latent score $y^* = \eta + \epsilon$ has its noise density centred on the linear predictor $\eta$. The thresholds $\theta_1 < \theta_2 < \theta_3$ cut the real line into four intervals, and each class probability is the area of the density over its interval. Drag the thresholds (or use their sliders); move $\eta$ to shift the whole density. Moving $\eta$ right moves mass towards the higher classes, which is the only way the inputs act." />
      }
      controls={
        <>
          <ParamSlider label="linear predictor η" value={eta} onChange={setEta} min={-5} max={5} step={0.1} />
          <ParamChoice label="link" value={link} onChange={setLink} options={LINKS} />
          {theta.map((t, j) => (
            <ParamSlider
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
      readout={probs.map((p, k) => (
        <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
      ))}
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
        <XYChart
          series={series}
          handles={handles}
          xRange={[-7, 7]}
          yRange={[0, link === 'logit' ? 0.27 : 0.42]}
          xLabel="latent score y*"
          yLabel="density"
          height={280}
          ariaLabel="Latent noise density split into class regions by thresholds"
        />
        <XYChart
          series={bars}
          xRange={[0.5, 4.5]}
          integerX
          yRange={[0, 1]}
          xLabel="class k"
          yLabel="probability"
          height={280}
          ariaLabel="Class probabilities"
        />
      </div>
    </Interactive>
  )
}
