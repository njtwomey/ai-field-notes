import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import { effectiveSampleSize, splitRhat } from '../../_shared/mcmc'

const CHAINS = 4
const STARTS = [-1, -0.33, 0.33, 1]

/**
 * Four chains, each an AR(1) process x_t = μ + φ (x_{t−1} − μ) + √(1 − φ²) ε_t with stationary distribution N(μ, 1).
 * The chains start at dispersed points, and the last one can be centred on a different mode μ = δ, standing in for a
 * chain trapped in a second mode.
 */
export function ChainDiagnostics() {
  const phi = useParam(0.9, { min: 0, max: 0.995, step: 0.005 })
  const draws = useParam(1000, { min: 100, max: 4000, step: 100 })
  const spread = useParam(8, { min: 0, max: 20, step: 0.5 })
  const offset = useParam(0, { min: 0, max: 3, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const run = useMemo(() => {
    const g = rng(seed.value)
    const noise = Math.sqrt(1 - phi.value ** 2)
    const chains = Array.from({ length: CHAINS }, (_, c) => {
      const mu = c === CHAINS - 1 ? offset.value : 0
      let x = STARTS[c] * spread.value
      const out: number[] = []
      for (let t = 0; t < draws.value; t++) {
        x = mu + phi.value * (x - mu) + noise * g.normal()
        out.push(x)
      }
      return out
    })
    const kept = chains.map((c) => c.slice(Math.floor(c.length / 2)))
    return {
      chains,
      rhatAll: splitRhat(chains),
      rhatKept: splitRhat(kept),
      ess: kept.reduce((a, c) => a + effectiveSampleSize(c), 0),
      keptCount: kept.reduce((a, c) => a + c.length, 0),
    }
  }, [phi.value, draws.value, spread.value, offset.value, seed.value])

  const series: XYSeries[] = useMemo(
    () =>
      run.chains.map((c, i) => ({
        name: `chain ${i + 1}`,
        type: 'line',
        x: c.map((_, t) => t + 1),
        y: c,
        slot: i,
      })),
    [run],
  )

  const tau = (1 + phi.value) / (1 - phi.value)

  return (
    <Interactive
      title="Trace plots, split R-hat and effective sample size"
      caption="Four chains whose stationary distribution is N(0, 1), each moving with autocorrelation φ from a dispersed start. The first half of each chain is treated as warm-up. With large starting spread, R-hat over all draws is far above 1 until the chains forget their starts. Raise φ: the chains mix slowly, the effective sample size falls towards (draws/2) × 4/τ, and R-hat on the kept half rises. Move chain 4 to another mode: every chain on its own looks stationary, but the chains disagree and R-hat flags it."
      controls={
        <>
          <ParamSlider label="autocorrelation φ" param={phi} />
          <ParamSlider label="draws per chain" param={draws} format={(v) => String(v)} />
          <ParamSlider label="starting spread" param={spread} />
          <ParamSlider label="chain 4 centred at δ" param={offset} />
          <ParamSlider label="random seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="split R-hat, all draws" value={formatNumber(run.rhatAll)} />
          <Readout label="split R-hat, second halves" value={formatNumber(run.rhatKept)} />
          <Readout label="ESS, second halves" value={formatNumber(run.ess)} />
          <Readout label="kept draws" value={run.keptCount} />
          <Readout label="τ in theory" value={formatNumber(tau)} />
        </>
      }
    >
      <XYChart series={series} xLabel="iteration" yLabel="x" height={300} />
    </Interactive>
  )
}
