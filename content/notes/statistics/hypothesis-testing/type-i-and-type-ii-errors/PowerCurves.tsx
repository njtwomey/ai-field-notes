import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf, normalQuantile } from '@/lib/math/special'

type Sides = 'two' | 'one'

/**
 * The z statistic under H₀ (centred at 0) and under H₁ (centred at δ√n/σ). Shaded: α, the rejection region under H₀,
 * and β, the acceptance region under H₁.
 */
export function PowerCurves() {
  const effectParam = useParam(0.4, { min: 0, max: 1.5, step: 0.01 })
  const [n, setN] = useState(25)
  const alphaParam = useParam(0.05, { min: 0.001, max: 0.2, step: 0.001 })
  const effect = effectParam.value
  const alpha = alphaParam.value
  const [sides, setSides] = useState<Sides>('two')

  const result = useMemo(() => {
    const shift = effect * Math.sqrt(n)
    const crit = normalQuantile(1 - (sides === 'two' ? alpha / 2 : alpha))
    const lo = Math.min(-4, shift - 4)
    const hi = Math.max(4, shift + 4)
    const xs = linspace(lo, hi, 400)
    const h0 = (x: number) => normalPdf(x)
    const h1 = (x: number) => normalPdf(x - shift)
    const region = (f: (x: number) => number, keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map(f) }
    }
    const reject = (x: number) => (sides === 'two' ? Math.abs(x) >= crit : x >= crit)
    const power = sides === 'two' ? 1 - normalCdf(crit - shift) + normalCdf(-crit - shift) : 1 - normalCdf(crit - shift)
    // n giving 80% power for this effect: n = ((z_crit + z_0.8) / δ)².
    const needed = effect > 0 ? Math.ceil(((crit + normalQuantile(0.8)) / effect) ** 2) : Infinity
    const tail = (keep: (x: number) => boolean) =>
      sides === 'two'
        ? [region(h0, (x) => x <= -crit && keep(x)), region(h0, (x) => x >= crit && keep(x))]
        : [region(h0, (x) => x >= crit && keep(x))]
    const series: XYSeries[] = [
      { name: 'Z under H₀', type: 'line', x: xs, y: xs.map(h0), slot: 0 },
      { name: 'Z under H₁', type: 'line', x: xs, y: xs.map(h1), slot: 1 },
      ...tail(() => true).map((r): XYSeries => ({
        name: 'α: false rejection',
        type: 'line',
        ...r,
        slot: 0,
        area: true,
      })),
      { name: 'β: missed effect', type: 'line', ...region(h1, (x) => !reject(x)), slot: 1, area: true },
    ]
    return { series, power, crit, needed, lo, hi, shift }
  }, [effect, n, alpha, sides])

  // The cut-off maps back to α through the tail area beyond it; the H₁ mean δ√n maps back to the effect δ.
  const handles: Handle[] = [
    {
      kind: 'x',
      at: result.crit,
      label: 'critical value',
      onDrag: (c) => alphaParam.set(sides === 'two' ? 2 * (1 - normalCdf(Math.abs(c))) : 1 - normalCdf(c)),
    },
    { kind: 'x', at: result.shift, label: 'H₁ mean', onDrag: (m) => effectParam.set(m / Math.sqrt(n)) },
  ]

  return (
    <Interactive
      title="α, β and power"
      caption="The blue curve is the test statistic when there is no effect; the orange curve is the same statistic when the effect is real. Shaded blue: rejections that are false (α). Shaded orange: real effects the test misses (β). Power is 1 − β. Raise n or the effect: the curves separate and β shrinks. Lower α: the cut-off moves out and β grows. Drag the dashed lines to move the cut-off, which sets α, or the mean under H₁, which sets the effect."
      controls={
        <>
          <ParamSlider label="effect δ (standard deviations)" param={effectParam} />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={2} max={200} step={1} />
          <ParamSlider label="significance level α" param={alphaParam} />
          <ParamChoice
            label="test"
            value={sides}
            onChange={setSides}
            options={[
              { value: 'two', label: 'two-sided' },
              { value: 'one', label: 'one-sided' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="critical value" value={formatNumber(result.crit)} />
          <Readout label="power 1 − β" value={`${(100 * result.power).toFixed(1)}%`} />
          <Readout label="β" value={`${(100 * (1 - result.power)).toFixed(1)}%`} />
          <Readout label="n for 80% power" value={Number.isFinite(result.needed) ? result.needed : '—'} />
        </>
      }
    >
      <XYChart
        height={300}
        series={result.series}
        xRange={[result.lo, result.hi]}
        yRange={[0, 0.45]}
        xLabel="z"
        yLabel="density"
        handles={handles}
      />
    </Interactive>
  )
}
