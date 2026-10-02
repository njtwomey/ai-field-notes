import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { useClassColors } from '../_shared/classColor'
import { fitGpOrdinal, gpOrdinalData } from '../_shared/laplace'

const XS = linspace(-3.2, 3.2, 129)
const DATA = gpOrdinalData()
const GAP = 0.1

/**
 * Gaussian-process ordinal regression on 30 points with four classes, fitted by the Laplace approximation. The
 * thresholds on the latent axis are draggable; the readout shows how the approximate evidence responds.
 */
export function GpOrdinal() {
  const [lengthscale, setLengthscale] = useState(1)
  const [sigma, setSigma] = useState(0.3)
  const [b, setB] = useState([-1, 0, 1])

  const setThreshold = (j: number) => (v: number) =>
    setB((th) => {
      const lo = j > 0 ? th[j - 1] + GAP : -3.5
      const hi = j < th.length - 1 ? th[j + 1] - GAP : 3.5
      const next = th.slice()
      next[j] = Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100
      return next
    })

  const colors = useClassColors(4)
  const [b0, b1, b2] = b
  const result = useMemo(() => {
    const fit = fitGpOrdinal(DATA.x, DATA.y, [b0, b1, b2], lengthscale, sigma)
    const latent = XS.map(fit.latent)
    const sd = latent.map((l) => 2 * Math.sqrt(l.variance))
    const latentSeries: XYSeries[] = [
      { name: 'posterior mean of f', type: 'line', x: XS, y: latent.map((l) => l.mean), emphasis: true },
      { name: '± 2 sd', type: 'line', x: XS, y: latent.map((l, i) => l.mean + sd[i]), muted: true, dashed: true },
      { name: '− 2 sd', type: 'line', x: XS, y: latent.map((l, i) => l.mean - sd[i]), muted: true, dashed: true },
      {
        name: 'mode at a training point',
        type: 'scatter',
        x: DATA.x,
        y: fit.mode,
        pointColors: DATA.y.map((y) => colors[y]),
      },
    ]
    const probs = XS.map(fit.probs)
    const probSeries: XYSeries[] = [0, 1, 2, 3].map((k) => ({
      name: `P(y = ${k + 1} | x)`,
      type: 'line',
      x: XS,
      y: probs.map((p) => p[k]),
      color: colors[k],
    }))
    const correct = DATA.x.filter((v, i) => {
      const p = fit.probs(v)
      return p.indexOf(Math.max(...p)) === DATA.y[i]
    }).length
    return { latentSeries, probSeries, evidence: fit.logEvidence, accuracy: correct / DATA.x.length }
  }, [b0, b1, b2, lengthscale, sigma, colors])

  const handles: Handle[] = b.map((t, j) => ({ kind: 'y', at: t, label: `b${j + 1}`, onDrag: setThreshold(j) }))

  return (
    <Interactive
      title="Gaussian-process ordinal regression"
      caption={
        <MathText text="Top: the posterior over the latent function $f$ under a squared-exponential kernel, with the thresholds $b_1 < b_2 < b_3$ as horizontal lines. Markers show the posterior mode $\hat f(x_i)$ at each training input, coloured by its observed class; a good fit places each marker between its class's thresholds. Bottom: the predictive class probabilities, which widen where the latent posterior is uncertain. Drag a threshold to move it and watch the approximate log evidence, which is the quantity Chu and Ghahramani maximise to set the thresholds, the lengthscale and the noise." />
      }
      controls={
        <>
          <ParamSlider
            label="lengthscale ℓ"
            value={lengthscale}
            onChange={setLengthscale}
            min={0.2}
            max={3}
            step={0.05}
          />
          <ParamSlider label="noise σ" value={sigma} onChange={setSigma} min={0.05} max={1} step={0.01} />
        </>
      }
      readout={
        <>
          <Readout label="log evidence (Laplace)" value={formatNumber(result.evidence)} />
          <Readout label="training accuracy" value={formatNumber(result.accuracy)} />
          <Readout label="thresholds" value={b.map((t) => formatNumber(t)).join(', ')} />
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <XYChart
          series={result.latentSeries}
          handles={handles}
          xRange={[-3.2, 3.2]}
          yRange={[-3.5, 3.5]}
          xLabel="input x"
          yLabel="latent f"
          height={280}
          ariaLabel="Latent posterior with draggable thresholds"
        />
        <XYChart
          series={result.probSeries}
          xRange={[-3.2, 3.2]}
          yRange={[0, 1]}
          xLabel="input x"
          yLabel="class probability"
          height={220}
          ariaLabel="Predictive class probabilities"
        />
      </div>
    </Interactive>
  )
}
