import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'
import { effectiveSampleSize, splitRhat } from '../../_shared/mcmc'

/** Trace points drawn per chain; longer chains are thinned for drawing only. */
const TRACE_POINTS = 400

/**
 * Several chains, each an AR(1) process x_t = μ + φ (x_{t−1} − μ) + √(1 − φ²) ε_t with stationary distribution
 * N(μ, 1). The chains start at points spread evenly across [−s, s], and the last one can be centred on a different
 * mode μ = δ, standing in for a chain trapped in a second mode.
 */
export function ChainDiagnostics() {
  const chains = useParam(4, { min: 1, max: 20, step: 1 })
  const phi = useParam(0.9, { min: 0, max: 0.995, step: 0.005 })
  const draws = useParam(1000, { min: 100, max: 4000, step: 100 })
  const spread = useParam(8, { min: 0, max: 20, step: 0.5 })
  const offset = useParam(0, { min: 0, max: 3, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const run = useMemo(() => {
    const m = chains.value
    const noise = Math.sqrt(1 - phi.value ** 2)
    // Each chain has its own random stream, so adding a chain leaves the others unchanged.
    const all = Array.from({ length: m }, (_, c) => {
      const g = rng(seed.value * 1000 + c)
      const mu = c === m - 1 ? offset.value : 0
      let x = (m === 1 ? 1 : -1 + (2 * c) / (m - 1)) * spread.value
      const out: number[] = []
      for (let t = 0; t < draws.value; t++) {
        x = mu + phi.value * (x - mu) + noise * g.normal()
        out.push(x)
      }
      return out
    })
    const kept = all.map((c) => c.slice(Math.floor(c.length / 2)))
    return {
      chains: all,
      rhatAll: splitRhat(all),
      rhatKept: splitRhat(kept),
      ess: kept.reduce((a, c) => a + effectiveSampleSize(c), 0),
      keptCount: kept.reduce((a, c) => a + c.length, 0),
    }
  }, [chains.value, phi.value, draws.value, spread.value, offset.value, seed.value])

  const series: XYSeries[] = useMemo(() => {
    const many = run.chains.length > 1
    const stride = Math.max(1, Math.ceil(draws.value / TRACE_POINTS))
    return run.chains.map((c, i): XYSeries => {
      const idx: number[] = []
      for (let t = 0; t < c.length; t += stride) idx.push(t)
      const last = i === run.chains.length - 1
      return {
        name: last ? (many ? 'last chain (centred at δ)' : 'chain (centred at δ)') : 'other chains',
        type: 'line',
        x: idx.map((t) => t + 1),
        y: idx.map((t) => c[t]),
        slot: last ? 1 : 0,
        thin: many,
      }
    })
  }, [run, draws.value])

  const tau = (1 + phi.value) / (1 - phi.value)

  return (
    <Interactive
      title="Trace plots, split R-hat and effective sample size"
      caption="Several chains whose stationary distribution is N(0, 1), each with its own random stream, moving with autocorrelation φ from starts spread evenly across ± the starting spread. Each chain is a light line; the chains slider sets how many run. The first half of each chain is treated as warm-up, and R-hat and the effective sample size (summed over chains) use every chain shown. With large starting spread, R-hat over all draws is far above 1 until the chains forget their starts. Raise φ: the chains mix slowly, the effective sample size falls towards (draws/2) × chains/τ, and R-hat on the kept half rises. Move the last chain to another mode: every chain on its own looks stationary, but the chains disagree and R-hat flags it."
      controls={
        <>
          <ParamSlider label="chains" param={chains} format={(v) => String(v)} withArrows />
          <ParamSlider label="autocorrelation φ" param={phi} />
          <ParamSlider label="draws per chain" param={draws} format={(v) => String(v)} />
          <ParamSlider label="starting spread" param={spread} />
          <ParamSlider label="last chain centred at δ" param={offset} />
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
