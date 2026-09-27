import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from '@/components/viz'
import { linspace, sigmoid } from '@/lib/math'

// A small nonlinear model with an interaction term: f(x) = σ(2 x₁ + x₂ + 1.5 x₁ x₂ − 1).
const score = (x1: number, x2: number) => 2 * x1 + x2 + 1.5 * x1 * x2 - 1
const f = (x1: number, x2: number) => sigmoid(score(x1, x2))
function grad(x1: number, x2: number): [number, number] {
  const p = f(x1, x2)
  const s = p * (1 - p)
  return [s * (2 + 1.5 * x2), s * (1 + 1.5 * x1)]
}

const LO = -2
const HI = 2
const AXIS = linspace(LO, HI, 41)
const GRID = AXIS.map((y) => AXIS.map((x) => f(x, y)))
const RANGE: [number, number] = [0, 1]

/**
 * Integrated gradients along the straight line from a baseline to an input, approximated by a midpoint Riemann sum.
 * The input and the baseline are draggable; the readouts compare the attributions' sum with f(x) − f(baseline).
 */
export function IntegratedGradients() {
  const x1 = useParam(1.5, { min: LO, max: HI, step: 0.05 })
  const x2 = useParam(1, { min: LO, max: HI, step: 0.05 })
  const b1 = useParam(0, { min: LO, max: HI, step: 0.05 })
  const b2 = useParam(0, { min: LO, max: HI, step: 0.05 })
  const steps = useParam(8, { min: 1, max: 64, step: 1 })

  const ig = useMemo(() => {
    const d1 = x1.value - b1.value
    const d2 = x2.value - b2.value
    let s1 = 0
    let s2 = 0
    const px: number[] = []
    const py: number[] = []
    for (let k = 0; k < steps.value; k++) {
      const a = (k + 0.5) / steps.value
      const p1 = b1.value + a * d1
      const p2 = b2.value + a * d2
      const [g1, g2] = grad(p1, p2)
      s1 += g1
      s2 += g2
      px.push(p1)
      py.push(p2)
    }
    return { a1: (d1 * s1) / steps.value, a2: (d2 * s2) / steps.value, px, py }
  }, [x1.value, x2.value, b1.value, b2.value, steps.value])

  const [g1, g2] = grad(x1.value, x2.value)
  const gxi1 = (x1.value - b1.value) * g1
  const gxi2 = (x2.value - b2.value) * g2
  const delta = f(x1.value, x2.value) - f(b1.value, b2.value)

  const overlay: HeatmapOverlay[] = [
    { name: 'path', type: 'line', x: [b1.value, x1.value], y: [b2.value, x2.value] },
    { name: 'Riemann points', type: 'scatter', x: ig.px, y: ig.py },
  ]
  const handles: Handle[] = [
    { kind: 'point', at: [x1.value, x2.value], label: 'input x', onDrag: ([a, b]) => (x1.set(a), x2.set(b)) },
    { kind: 'point', at: [b1.value, b2.value], label: 'baseline', onDrag: ([a, b]) => (b1.set(a), b2.set(b)) },
  ]

  return (
    <Interactive
      title="Integrated gradients on a two-feature model"
      caption="The background is the model output f(x) = σ(2x₁ + x₂ + 1.5x₁x₂ − 1). Drag the input and the baseline. Integrated gradients average the gradient at the Riemann points along the straight path and multiply by the displacement. Their sum approaches f(x) − f(baseline) as the number of steps grows. Gradient × (x − baseline) uses only the gradient at the input, so it misses most of the change when the input sits where the sigmoid has flattened."
      controls={
        <>
          <ParamSlider label="Riemann steps m" param={steps} format={(v) => String(v)} />
          <ParamSlider label="input x₁" param={x1} />
          <ParamSlider label="input x₂" param={x2} />
          <ParamSlider label="baseline x′₁" param={b1} />
          <ParamSlider label="baseline x′₂" param={b2} />
        </>
      }
      readout={
        <>
          <Readout label="f(x) − f(x′)" value={formatNumber(delta)} />
          <Readout label="IG₁, IG₂" value={`${formatNumber(ig.a1)}, ${formatNumber(ig.a2)}`} />
          <Readout label="IG₁ + IG₂" value={formatNumber(ig.a1 + ig.a2)} />
          <Readout label="gradient × (x − x′), summed" value={formatNumber(gxi1 + gxi2)} />
        </>
      }
    >
      <Heatmap
        x={AXIS}
        y={AXIS}
        z={GRID}
        range={RANGE}
        scale="sequential"
        xLabel="x₁"
        yLabel="x₂"
        valueLabel="f(x)"
        overlay={overlay}
        handles={handles}
        ariaLabel="Heatmap of a two-feature model with the straight path from a baseline to an input"
      />
    </Interactive>
  )
}
