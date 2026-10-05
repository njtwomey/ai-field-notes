import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalCdf, normalPdf, normalQuantile } from 'aifn-compute/numerics/special'

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
  const state = useFigureState({
    effect: float(0.1, { min: 0.02, max: 0.4, step: 0.01, label: 'true effect θ' }),
    n: int(200, { min: 20, max: 2000, step: 10, label: 'users per arm n' }),
    alpha: choice(ALPHAS, '0.05', { label: 'α' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })

  const s = Math.sqrt(2 / state.n)
  const exact = retrodesign(state.effect, s, Number(state.alpha))

  // One standard-normal draw per replication, rescaled on every change, so moving a slider moves the same experiments.
  const draws = useMemo(() => {
    const g = stream(state.seed)
    return Array.from({ length: EXPERIMENTS }, () => normal(g))
  }, [state.seed])

  const r = useMemo(() => {
    const estimates = draws.map((e) => state.effect + s * e)
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
      ] satisfies SeriesSpec[],
      power: significant.length / EXPERIMENTS,
      typeS: significant.length ? significant.filter((t) => t < 0).length / significant.length : NaN,
      exaggeration: sigMean / state.effect,
      sigMean,
    }
  }, [draws, state.effect, s, exact.z])

  const series: SeriesSpec[] = [
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

  const xAxis = useAxis({ label: 'estimated effect θ̂', hold: 'union' })
  const yAxis = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="Significant estimates overstate the effect"
      state={state}
      caption="4,000 replications of the same A/B test with n users per arm. Each bar counts estimates of the difference in means (in units of the outcome's standard deviation). Only the orange tails reach significance. When power is low, those tails sit far from the true effect: the average significant estimate is several times too large, and some point the wrong way. Drag the true effect or raise n to watch the exaggeration shrink towards 1."

      readouts={
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
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
        <Handle {...state.handle('effect', { label: 'true effect' })} />
      </Plot>
    </Figure>
  )
}
