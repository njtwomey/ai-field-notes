import { useMemo, useState } from 'react'
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
import { tPower } from '@/lib/math/tests'

/** Power of the pooled two-sample t-test against a standardised difference d, as the total N is split n₁ + n₂. */
export function Allocation() {
  const [total, setTotal] = useState(40)
  const [d, setD] = useState(0.8)
  const [alpha, setAlpha] = useState(0.05)
  const n1 = useParam(10, { min: 2, max: 198, step: 1 })
  const split = Math.min(Math.max(n1.value, 2), total - 2)

  const result = useMemo(() => {
    const power = (a: number) => tPower(d / Math.sqrt(1 / a + 1 / (total - a)), total - 2, alpha)
    const xs = Array.from({ length: total - 3 }, (_, i) => i + 2)
    const series: XYSeries[] = [{ name: 'power', type: 'line', x: xs, y: xs.map(power), slot: 0 }]
    return { series, power }
  }, [total, d, alpha])

  const handles: Handle[] = [{ kind: 'x', at: split, label: 'n₁', onDrag: (x) => n1.set(Math.round(x)) }]

  return (
    <Interactive
      title="Unequal groups waste observations"
      caption="Power of the pooled t-test for a fixed total number of observations, as a function of how many go to group 1. The standard error of the difference, σ√(1/n₁ + 1/n₂), is smallest for equal groups, so power peaks there. Mild imbalance costs little; strong imbalance costs a lot. Drag the line labelled n₁, or use its slider, to change the split."
      controls={
        <>
          <ParamSlider label="observations in group 1, n₁" param={n1} />
          <ParamSlider
            label="total N"
            value={total}
            onChange={(v) => {
              setTotal(v)
              if (n1.value > v - 2) n1.set(v - 2)
            }}
            min={10}
            max={200}
            step={2}
          />
          <ParamSlider label="true effect d" value={d} onChange={setD} min={0.1} max={2} step={0.05} />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.1}
            step={0.005}
          />
        </>
      }
      readout={
        <>
          <Readout label="split" value={`${split} + ${total - split}`} />
          <Readout label="power" value={formatNumber(result.power(split))} />
          <Readout label="power, equal split" value={formatNumber(result.power(Math.floor(total / 2)))} />
        </>
      }
    >
      <XYChart
        height={280}
        series={result.series}
        xRange={[0, total]}
        yRange={[0, 1]}
        xLabel="observations in group 1"
        yLabel="power"
        handles={handles}
      />
    </Interactive>
  )
}
