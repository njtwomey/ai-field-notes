import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { normalCdf } from '@/lib/math/special'

/**
 * Explore-then-commit on two unit-variance Gaussian arms with gap Δ: the exact expected regret
 * mΔ + (T − 2m)Δ Φ(−Δ√(m/2)) and the Hoeffding-style bound mΔ + (T − 2m)Δ exp(−mΔ²/4), as functions of the number
 * m of exploration pulls per arm.
 */
export function EtcTradeoff() {
  const gap = useParam(0.2, { min: 0.02, max: 1, step: 0.01 })
  const logT = useParam(4, { min: 2, max: 5, step: 0.1 })
  const T = Math.round(10 ** logT.value)
  const mMax = Math.floor(T / 2)
  const m = useParam(50, { min: 1, max: mMax, step: 1 })
  const mNow = Math.min(m.value, mMax)

  const curves = useMemo(() => {
    const d = gap.value
    const ms: number[] = []
    const step = Math.max(1, Math.floor(mMax / 400))
    for (let v = 1; v <= mMax; v += step) ms.push(v)
    const exact = (v: number) => v * d + (T - 2 * v) * d * normalCdf(-d * Math.sqrt(v / 2))
    const bound = (v: number) => v * d + (T - 2 * v) * d * Math.exp((-v * d * d) / 4)
    let best = 1
    for (let v = 1; v <= mMax; v++) if (exact(v) < exact(best)) best = v
    const formula = Math.min(mMax, Math.max(1, Math.ceil((4 / (d * d)) * Math.log((T * d * d) / 4))))
    return { ms, exact, bound, best, formula }
  }, [gap.value, T, mMax])

  const series: XYSeries[] = [
    { name: 'exact expected regret', type: 'line', x: curves.ms, y: curves.ms.map(curves.exact), slot: 0 },
    { name: 'upper bound', type: 'line', x: curves.ms, y: curves.ms.map(curves.bound), slot: 1, dashed: true },
    {
      name: 'exploration cost mΔ',
      type: 'line',
      x: [0, mMax],
      y: [0, mMax * gap.value],
      muted: true,
    },
    { name: 'chosen m', type: 'scatter', x: [mNow], y: [curves.exact(mNow)], emphasis: true },
  ]
  const wrong = normalCdf(-gap.value * Math.sqrt(mNow / 2))

  return (
    <Interactive
      title="How long to explore"
      caption="Two arms with unit-variance Gaussian rewards and gap Δ. Explore-then-commit pulls each arm m times, then commits for the remaining T − 2m rounds to the arm with the higher sample mean. Drag along the chart, or use the slider, to set m. Too little exploration commits to the wrong arm too often; too much wastes mΔ on the worse arm. The dashed curve is the bound used in the analysis, whose minimiser is the formula m."
      controls={
        <>
          <ParamSlider label="gap Δ" param={gap} />
          <ParamSlider label="horizon T" param={logT} format={(v) => String(Math.round(10 ** v))} />
          <ParamSlider label="exploration pulls per arm m" param={m} format={() => String(mNow)} />
        </>
      }
      readout={
        <>
          <Readout label="P(commit to the wrong arm)" value={formatNumber(wrong)} />
          <Readout label="regret at m" value={formatNumber(curves.exact(mNow))} />
          <Readout label="best m" value={`${curves.best} (regret ${formatNumber(curves.exact(curves.best))})`} />
          <Readout
            label="formula m = ⌈4 ln(TΔ²/4)/Δ²⌉"
            value={`${curves.formula} (regret ${formatNumber(curves.exact(curves.formula))})`}
          />
          <Readout label="never exploring, TΔ/2" value={formatNumber((T * gap.value) / 2)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="exploration pulls per arm m"
        yLabel="expected regret"
        xRange={[0, mMax]}
        yRange={[0, undefined]}
        handles={[{ kind: 'x', at: mNow, onDrag: (x) => m.set(Math.max(1, Math.round(x))) }]}
      />
    </Interactive>
  )
}
