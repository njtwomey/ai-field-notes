import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

const log2 = (x: number) => Math.log(x) / Math.LN2
/** The true distribution: four outcomes with probabilities 1/2, 1/4, 1/8, 1/8, so H(p) = 1.75 bits. */
const P = [0.5, 0.25, 0.125, 0.125]
const OUTCOMES = [1, 2, 3, 4]
const H_P = -P.reduce((s, p) => s + p * log2(p), 0)

/** Model q = p tempered by T: q ∝ p^(1/T). T = 1 is exact, large T flattens towards uniform, small T sharpens. */
function model(t: number): number[] {
  const w = P.map((p) => p ** (1 / t))
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / total)
}
const crossEntropy = (q: number[]) => -P.reduce((s, p, i) => s + p * log2(q[i]), 0)

const T_GRID = Array.from({ length: 121 }, (_, i) => 0.25 * Math.pow(16, i / 120))
const CURVE = T_GRID.map((t) => crossEntropy(model(t)))

/** Cross-entropy of a tempered model against the true distribution, split into entropy plus KL divergence. */
export function CrossEntropyGap() {
  const temperature = useParam(2, { min: 0.25, max: 4, step: 0.05 })
  const q = useMemo(() => model(temperature.value), [temperature.value])
  const hpq = crossEntropy(q)

  const bars: XYSeries[] = [
    { name: 'model q', type: 'bar', x: OUTCOMES, y: q, slot: 0 },
    { name: 'true p', type: 'scatter', x: OUTCOMES, y: P, emphasis: true },
  ]
  const curve: XYSeries[] = [
    { name: 'cross-entropy H(p, q)', type: 'line', x: T_GRID, y: CURVE, slot: 0 },
    { name: 'entropy H(p)', type: 'line', x: [0.25, 4], y: [H_P, H_P], dashed: true, slot: 1 },
    { name: 'current model', type: 'scatter', x: [temperature.value], y: [hpq], emphasis: true },
  ]
  // The temperature is the horizontal position on the curve, so dragging along the axis sets it.
  const handles: Handle[] = [{ kind: 'x', at: temperature.value, label: 'T', onDrag: (x) => temperature.set(x) }]

  return (
    <Interactive
      title="Cross-entropy is entropy plus KL divergence"
      caption="The true distribution p has probabilities 1/2, 1/4, 1/8, 1/8, so its entropy is 1.75 bits. The model q is p tempered by T, q ∝ p^(1/T): T = 1 is exact, larger T flattens q towards uniform and smaller T sharpens it. Left: p (diamonds) against q (bars). Right: the cross-entropy H(p, q) against T. It never falls below H(p), and the gap is KL(p ‖ q). Drag T or use the slider."
      controls={<ParamSlider label="temperature T" param={temperature} />}
      readout={
        <>
          <Readout label="H(p)" value={`${formatNumber(H_P)} bits`} />
          <Readout label="KL(p ‖ q)" value={`${formatNumber(hpq - H_P)} bits`} />
          <Readout label="H(p, q)" value={`${formatNumber(hpq)} bits`} />
          <Readout label="perplexity of q" value={formatNumber(2 ** hpq)} />
          <Readout label="perplexity of p" value={formatNumber(2 ** H_P)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={bars} xLabel="outcome" yLabel="probability" xRange={[0.5, 4.5]} yRange={[0, 1]} height={300} />
        <XYChart
          series={curve}
          xLabel="temperature T"
          yLabel="bits"
          xRange={[0.25, 4]}
          yRange={[1.5, 3.2]}
          handles={handles}
          height={300}
        />
      </div>
    </Interactive>
  )
}
