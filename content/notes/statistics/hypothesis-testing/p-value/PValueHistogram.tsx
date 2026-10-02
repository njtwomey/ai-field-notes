import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { normalCdf, normalQuantile } from '@/lib/math/special'

const STUDIES = 2000
const BINS = 20

/**
 * Many repeats of a two-sided z-test with effect size δ (in standard deviations) and n observations. The z statistic
 * is drawn from its exact sampling distribution, N(δ√n, 1).
 */
export function PValueHistogram() {
  const [effect, setEffect] = useState(0)
  const [n, setN] = useState(30)
  const alphaParam = useParam(0.05, { min: 0.005, max: 0.2, step: 0.005 })
  const alpha = alphaParam.value
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const r = rng(seed)
    const shift = effect * Math.sqrt(n)
    const counts = new Array(BINS).fill(0)
    let below = 0
    for (let i = 0; i < STUDIES; i++) {
      const z = shift + r.normal()
      const p = 2 * (1 - normalCdf(Math.abs(z)))
      counts[Math.min(BINS - 1, Math.floor(p * BINS))]++
      if (p < alpha) below++
    }
    const crit = normalQuantile(1 - alpha / 2)
    const power = 1 - normalCdf(crit - shift) + normalCdf(-crit - shift)
    const width = 1 / BINS
    const series: XYSeries[] = [
      {
        name: 'p-values',
        type: 'bar',
        x: counts.map((_, i) => (i + 0.5) * width),
        y: counts.map((c) => c / (STUDIES * width)),
        slot: 0,
      },
      { name: 'uniform density', type: 'line', x: [0, 1], y: [1, 1], slot: 2, dashed: true },
    ]
    return { series, share: below / STUDIES, power }
  }, [effect, n, alpha, seed])
  const handles: Handle[] = [{ kind: 'x', at: alpha, label: 'α', onDrag: alphaParam.set }]

  return (
    <Interactive
      title="2,000 studies, one p-value each"
      caption="With no effect (δ = 0) the p-values are uniform: about α of them fall below α by chance. Add an effect and they pile up near zero; the share below α is then the test's power. The dashed line is α: drag it to change the significance level."
      controls={
        <>
          <ParamSlider
            label="true effect δ (standard deviations)"
            value={effect}
            onChange={setEffect}
            min={0}
            max={1}
            step={0.01}
          />
          <ParamSlider label="observations per study n" value={n} onChange={setN} min={5} max={200} step={1} />
          <ParamSlider label="significance level α" param={alphaParam} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>Rerun</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="share with p < α" value={`${(100 * result.share).toFixed(1)}%`} />
          <Readout
            label={effect === 0 ? 'expected (α)' : 'theoretical power'}
            value={`${(100 * (effect === 0 ? alpha : result.power)).toFixed(1)}%`}
          />
          <Readout label="shift of z, δ√n" value={formatNumber(effect * Math.sqrt(n))} />
        </>
      }
    >
      <XYChart
        height={280}
        series={result.series}
        xRange={[0, 1]}
        yRange={[0, undefined]}
        xLabel="p-value"
        yLabel="density"
        handles={handles}
      />
    </Interactive>
  )
}
