import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { binomialUpper } from '@/lib/math/tests'

const N_MAX = 100

/** Attained size of the exact one-sided binomial test of a fair coin, H₁: p > 1/2, for every n up to N_MAX. */
export function SizeStaircase() {
  const [alpha, setAlpha] = useState(0.05)
  const [n, setN] = useState(20)

  const result = useMemo(() => {
    const ns = Array.from({ length: N_MAX - 4 }, (_, i) => i + 5)
    const sizes = ns.map((m) => {
      let c = 0
      while (binomialUpper(c, m, 0.5) > alpha) c++
      return binomialUpper(c, m, 0.5)
    })
    const series: XYSeries[] = [
      { name: 'attained size', type: 'line', x: ns, y: sizes, slot: 0 },
      { name: 'nominal α', type: 'line', x: [5, N_MAX], y: [alpha, alpha], slot: 1, dashed: true },
      { name: 'selected n', type: 'scatter', x: [n], y: [sizes[n - 5]], emphasis: true },
    ]
    let c = 0
    while (binomialUpper(c, n, 0.5) > alpha) c++
    return { series, c, size: sizes[n - 5] }
  }, [alpha, n])

  return (
    <Interactive
      title="The size of an exact test is a staircase"
      caption="An exact test of a fair coin rejects when the number of heads reaches a critical count c. Only whole counts are possible, so the probability of rejecting a true null, the attained size, jumps with n and stays at or below the nominal level α (dashed)."
      controls={
        <>
          <ParamSlider label="nominal level α" value={alpha} onChange={setAlpha} min={0.01} max={0.1} step={0.005} />
          <ParamSlider label="tosses n" value={n} onChange={setN} min={5} max={N_MAX} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="critical count c" value={result.c} />
          <Readout label="attained size" value={formatNumber(result.size)} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="tosses n" yLabel="P(reject | H₀)" yRange={[0, undefined]} />
    </Interactive>
  )
}
