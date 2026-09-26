import { useState } from 'react'
import { Interactive, ParamSlider, XYChart } from '@/components/viz'
import { linspace, sigmoid } from '@/lib/math'

export function SigmoidExplorer() {
  const [w, setW] = useState(1)
  const [b, setB] = useState(0)
  const x = linspace(-6, 6, 121)
  return (
    <Interactive
      title="The logistic function"
      caption="The weight sets the steepness. The bias shifts the point where the probability crosses 0.5, at x = −b/w."
      controls={
        <>
          <ParamSlider label="weight w" value={w} onChange={setW} min={-4} max={4} step={0.1} />
          <ParamSlider label="bias b" value={b} onChange={setB} min={-4} max={4} step={0.1} />
        </>
      }
    >
      <XYChart
        height={260}
        yRange={[0, 1]}
        xLabel="x"
        yLabel="P(y = 1 | x)"
        series={[
          { name: 'σ(wx + b)', type: 'line', x, y: x.map((xi) => sigmoid(w * xi + b)) },
          { name: '0.5', type: 'line', x: [-6, 6], y: [0.5, 0.5], slot: 1, dashed: true },
        ]}
      />
    </Interactive>
  )
}
