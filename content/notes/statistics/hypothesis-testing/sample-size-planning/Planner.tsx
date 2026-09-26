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
import { normalQuantile } from '@/lib/math/special'
import { tPower } from '@/lib/math/tests'

const N_MIN = 5
const N_MAX = 500

/** Smallest n per group whose exact two-sample t-test power reaches `target` for effect d (binary search on n). */
function exactN(d: number, alpha: number, target: number): number | undefined {
  const power = (n: number) => tPower(d * Math.sqrt(n / 2), 2 * n - 2, alpha)
  if (d <= 0 || power(N_MAX * 4) < target) return undefined
  let lo = 2
  let hi = N_MAX * 4
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (power(mid) >= target) hi = mid
    else lo = mid + 1
  }
  return lo
}

/**
 * Minimum detectable effect of a two-sided two-sample comparison of means with n per group, in standard deviations:
 * MDE = (z₁₋α/₂ + z₁₋β)·√(2/n). The readouts also invert it for a target effect, by the same approximation and by
 * exact t-test power.
 */
export function Planner() {
  const [alpha, setAlpha] = useState(0.05)
  const [power, setPower] = useState(0.8)
  const [target, setTarget] = useState(0.5)
  const n = useParam(64, { min: N_MIN, max: N_MAX, step: 1 })

  const result = useMemo(() => {
    const k = normalQuantile(1 - alpha / 2) + normalQuantile(power)
    const ns = Array.from({ length: N_MAX - N_MIN + 1 }, (_, i) => i + N_MIN)
    const series: XYSeries[] = [
      { name: 'minimum detectable effect', type: 'line', x: ns, y: ns.map((m) => k * Math.sqrt(2 / m)), slot: 0 },
      { name: 'target effect', type: 'line', x: [N_MIN, N_MAX], y: [target, target], slot: 1, dashed: true },
    ]
    return { k, series, approx: Math.ceil((2 * k * k) / (target * target)), exact: exactN(target, alpha, power) }
  }, [alpha, power, target])

  const mde = result.k * Math.sqrt(2 / n.value)
  const handles: Handle[] = [{ kind: 'x', at: n.value, label: 'n', onDrag: (x) => n.set(Math.round(x)) }]

  return (
    <Interactive
      title="Minimum detectable effect against sample size"
      caption="The curve is the smallest standardised difference in means that a two-sided two-sample test detects with the chosen power, for n per group. It falls like 1/√n. Where it crosses the dashed target effect is the sample size needed. Drag the line labelled n, or use its slider, to read the effect detectable with that many observations."
      controls={
        <>
          <ParamSlider label="n per group" param={n} />
          <ParamSlider label="target effect d" value={target} onChange={setTarget} min={0.05} max={1.5} step={0.05} />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.005}
            max={0.2}
            step={0.005}
          />
          <ParamSlider label="power 1 − β" value={power} onChange={setPower} min={0.5} max={0.99} step={0.01} />
        </>
      }
      readout={
        <>
          <Readout label="MDE at this n" value={formatNumber(mde)} />
          <Readout label="n per group, normal approx." value={result.approx} />
          <Readout label="n per group, exact t" value={result.exact ?? `> ${N_MAX * 4}`} />
        </>
      }
    >
      <XYChart
        height={300}
        series={result.series}
        xRange={[0, N_MAX]}
        yRange={[0, 2]}
        xLabel="n per group"
        yLabel="effect d (standard deviations)"
        handles={handles}
      />
    </Interactive>
  )
}
