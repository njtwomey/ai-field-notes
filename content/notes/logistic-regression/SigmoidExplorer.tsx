import { Interactive, ParamSlider, XYChart, useParam, type Handle } from '@/components/viz'
import { linspace, sigmoid } from '@/lib/math'

export function SigmoidExplorer() {
  const w = useParam(1, { min: -4, max: 4, step: 0.1 })
  const b = useParam(0, { min: -4, max: 4, step: 0.1 })
  const x = linspace(-6, 6, 121)
  // The crossing point x = −b/w as a handle: dragging it sets b = −w·x. Undefined when w = 0 (no crossing). Shown
  // at the edge of the plot when the crossing lies beyond it.
  const crossing = Math.abs(w.value) > 1e-9 ? -b.value / w.value : undefined
  const handles: Handle[] | undefined =
    crossing === undefined
      ? undefined
      : [{ kind: 'x', at: Math.min(Math.max(crossing, -6), 6), label: 'P = 0.5', onDrag: (at) => b.set(-w.value * at) }]
  return (
    <Interactive
      title="The logistic function"
      caption="The weight sets the steepness. The bias shifts the point where the probability crosses 0.5, at x = −b/w. Drag the dashed line to move that point; the bias follows."
      controls={
        <>
          <ParamSlider label="weight w" param={w} />
          <ParamSlider label="bias b" param={b} />
        </>
      }
    >
      <XYChart
        height={260}
        xRange={[-6, 6]}
        yRange={[0, 1]}
        xLabel="x"
        yLabel="P(y = 1 | x)"
        handles={handles}
        series={[
          { name: 'σ(wx + b)', type: 'line', x, y: x.map((xi) => sigmoid(w.value * xi + b.value)) },
          { name: '0.5', type: 'line', x: [-6, 6], y: [0.5, 0.5], slot: 1, dashed: true },
        ]}
      />
    </Interactive>
  )
}
