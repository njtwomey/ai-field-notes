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
import { rng } from '@/lib/math'
import { normalCdf, normalPdf, normalQuantile } from '@/lib/math/special'

const EXPERIMENTS = 4000
/** Bins per critical distance z·s, so the significance threshold falls on a bin edge and no bin is split. */
const BINS_PER_CRIT = 5
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]

/** Exact power, type S rate and exaggeration ratio for an estimate θ̂ ~ N(θ, s²) tested two-sided at level α. */
function retrodesign(theta: number, s: number, alpha: number) {
  const z = normalQuantile(1 - alpha / 2)
  const d = theta / s
  const high = 1 - normalCdf(z - d)
  const low = normalCdf(-z - d)
  const power = high + low
  // E[|θ̂|; significant] = E[θ̂; θ̂ > zs] − E[θ̂; θ̂ < −zs], each a truncated-normal partial mean.
  const partial = s * (d * high + normalPdf(z - d)) + s * (-d * low + normalPdf(z + d))
  return { z, power, typeS: low / power, exaggeration: partial / power / theta }
}

/**
 * Many replications of one underpowered A/B test. Estimates that reach significance are the ones in the tails, so on
 * average they overstate the true effect, and some have the wrong sign.
 */
export function WinnersCurse() {
  const effect = useParam(0.1, { min: 0.02, max: 0.4, step: 0.01 })
  const n = useParam(200, { min: 20, max: 2000, step: 10 })
  const [alpha, setAlpha] = useState('0.05')
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const s = Math.sqrt(2 / n.value)
  const exact = retrodesign(effect.value, s, Number(alpha))

  // One standard-normal draw per replication, rescaled on every change, so moving a slider moves the same experiments.
  const draws = useMemo(() => {
    const g = rng(seed.value)
    return Array.from({ length: EXPERIMENTS }, () => g.normal())
  }, [seed.value])

  const r = useMemo(() => {
    const estimates = draws.map((e) => effect.value + s * e)
    const crit = exact.z * s
    const significant = estimates.filter((t) => Math.abs(t) >= crit)
    const width = crit / BINS_PER_CRIT
    const counts = new Map<number, number>()
    for (const t of estimates) {
      const bin = Math.floor(t / width)
      counts.set(bin, (counts.get(bin) ?? 0) + 1)
    }
    const bins = [...counts.keys()].sort((a, b) => a - b)
    const centre = (b: number) => (b + 0.5) * width
    const isSig = (b: number) => Math.abs(centre(b)) >= crit
    const density = (b: number) => (counts.get(b) ?? 0) / (EXPERIMENTS * width)
    const sigMean = significant.length ? significant.reduce((a, t) => a + Math.abs(t), 0) / significant.length : NaN
    return {
      bins,
      series: [
        {
          name: 'not significant',
          type: 'bar',
          x: bins.filter((b) => !isSig(b)).map(centre),
          y: bins.filter((b) => !isSig(b)).map(density),
          muted: true,
        },
        {
          name: 'significant',
          type: 'bar',
          x: bins.filter(isSig).map(centre),
          y: bins.filter(isSig).map(density),
          slot: 1,
        },
      ] satisfies XYSeries[],
      power: significant.length / EXPERIMENTS,
      typeS: significant.length ? significant.filter((t) => t < 0).length / significant.length : NaN,
      exaggeration: sigMean / effect.value,
      sigMean,
    }
  }, [draws, effect.value, s, exact.z])

  const handles: Handle[] = [{ kind: 'x', at: effect.value, label: 'true effect', onDrag: effect.set }]
  const series: XYSeries[] = [
    ...r.series,
    ...(Number.isFinite(r.sigMean)
      ? [
          {
            name: 'mean |estimate| when significant',
            type: 'line' as const,
            x: [r.sigMean, r.sigMean],
            y: [0, Math.max(...r.series.flatMap((b) => b.y)) * 1.05],
            slot: 2,
            dashed: true,
          },
        ]
      : []),
  ]

  return (
    <Interactive
      title="Significant estimates overstate the effect"
      caption="4,000 replications of the same A/B test with n users per arm. Each bar counts estimates of the difference in means (in units of the outcome's standard deviation). Only the orange tails reach significance. When power is low, those tails sit far from the true effect: the average significant estimate is several times too large, and some point the wrong way. Drag the true effect or raise n to watch the exaggeration shrink towards 1."
      controls={
        <>
          <ParamSlider label="true effect θ" param={effect} />
          <ParamSlider label="users per arm n" param={n} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
          <ParamSlider label="seed" param={seed} />
        </>
      }
      readout={
        <>
          <Readout label="standard error s" value={formatNumber(s)} />
          <Readout label="power" value={`${formatNumber(exact.power)} (sim ${formatNumber(r.power)})`} />
          <Readout label="type S rate" value={`${formatNumber(exact.typeS)} (sim ${formatNumber(r.typeS)})`} />
          <Readout
            label="exaggeration ratio"
            value={`${formatNumber(exact.exaggeration)} (sim ${formatNumber(r.exaggeration)})`}
          />
        </>
      }
    >
      <XYChart height={340} xLabel="estimated effect θ̂" yLabel="density" series={series} handles={handles} />
    </Interactive>
  )
}
