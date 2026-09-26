import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'

type Test = { p: number; effect: boolean }

/**
 * m one-sided z-tests: a share of them have a real effect (z ~ N(shift, 1)), the rest are true nulls (z ~ N(0, 1)).
 * Sorted p-values are compared with the Benjamini–Hochberg line q·i/m and the Bonferroni threshold q/m.
 */
export function StepUp() {
  const [m, setM] = useState(200)
  const [share, setShare] = useState(0.2)
  const [shift, setShift] = useState(3)
  const [seed, setSeed] = useState(1)
  const q = useParam(0.1, { min: 0.005, max: 0.5, step: 0.005 })

  const tests = useMemo((): Test[] => {
    const r = rng(seed)
    const effects = Math.round(share * m)
    return Array.from({ length: m }, (_, i) => {
      const effect = i < effects
      const z = (effect ? shift : 0) + r.normal()
      return { p: Math.max(1 - normalCdf(z), 1e-12), effect }
    }).sort((a, b) => a.p - b.p)
  }, [m, share, shift, seed])

  const result = useMemo(() => {
    // Benjamini–Hochberg: reject the k smallest, where k is the largest rank with p₍ₖ₎ ≤ q·k/m.
    let k = 0
    tests.forEach((t, i) => {
      if (t.p <= (q.value * (i + 1)) / m) k = i + 1
    })
    const bhFalse = tests.slice(0, k).filter((t) => !t.effect).length
    const bonf = tests.filter((t) => t.p <= q.value / m)
    const ranks = tests.map((_, i) => i + 1)
    const pick = (effect: boolean) => ({
      x: ranks.filter((_, i) => tests[i].effect === effect),
      y: tests.filter((t) => t.effect === effect).map((t) => t.p),
    })
    const series: XYSeries[] = [
      { name: 'true null', type: 'scatter', ...pick(false), slot: 0 },
      { name: 'real effect', type: 'scatter', ...pick(true), slot: 1 },
      { name: 'BH line q·i/m', type: 'line', x: ranks, y: ranks.map((i) => (q.value * i) / m), slot: 2 },
      { name: 'Bonferroni q/m', type: 'line', x: [1, m], y: [q.value / m, q.value / m], slot: 3, dashed: true },
    ]
    return {
      series,
      k,
      bhFalse,
      bonfCount: bonf.length,
      bonfFalse: bonf.filter((t) => !t.effect).length,
    }
  }, [tests, q.value, m])

  const handles: Handle[] = [{ kind: 'point', at: [m, q.value], label: 'q', onDrag: ([, y]) => q.set(y) }]
  const fdp = result.k ? result.bhFalse / result.k : 0

  return (
    <Interactive
      title="Benjamini–Hochberg on sorted p-values"
      caption="The m p-values are sorted and plotted against their rank on a log scale. Benjamini–Hochberg rejects every test up to the last one below the line q·i/m. Bonferroni rejects only those below q/m. Drag the end of the BH line, or use the q slider, to change the target false discovery rate."
      controls={
        <>
          <ParamSlider label="target level q" param={q} />
          <ParamSlider label="tests m" value={m} onChange={setM} min={20} max={1000} step={10} />
          <ParamSlider label="share with a real effect" value={share} onChange={setShare} min={0} max={1} step={0.05} />
          <ParamSlider label="effect size (z shift)" value={shift} onChange={setShift} min={0} max={5} step={0.1} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New data</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="BH rejections" value={`${result.k} (${result.bhFalse} false)`} />
          <Readout label="false discovery proportion" value={`${(100 * fdp).toFixed(1)}%`} />
          <Readout label="Bonferroni rejections" value={`${result.bonfCount} (${result.bonfFalse} false)`} />
        </>
      }
    >
      <XYChart
        height={320}
        series={result.series}
        yLog
        yRange={[undefined, 1]}
        xLabel="rank i"
        yLabel="p-value"
        handles={handles}
      />
    </Interactive>
  )
}
