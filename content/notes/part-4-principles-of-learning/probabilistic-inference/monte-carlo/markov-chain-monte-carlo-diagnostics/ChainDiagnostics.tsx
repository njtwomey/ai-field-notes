import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { effectiveSampleSize, splitRhat } from '../../_shared/mcmc'
import { normal, stream } from 'aifn/foundation/random'

/** Trace points drawn per chain; longer chains are thinned for drawing only. */
const TRACE_POINTS = 400

/**
 * Several chains, each an AR(1) process x_t = μ + φ (x_{t−1} − μ) + √(1 − φ²) ε_t with stationary distribution
 * N(μ, 1). The chains start at points spread evenly across [−s, s], and the last one can be centred on a different
 * mode μ = δ, standing in for a chain trapped in a second mode.
 */
export function ChainDiagnostics() {
  const state = useFigureState({
    chains: int(4, { min: 1, max: 20, step: 1, label: 'chains', format: (v) => String(v) }),
    phi: float(0.9, { min: 0, max: 0.995, step: 0.005, label: 'autocorrelation φ' }),
    draws: int(1000, { min: 100, max: 4000, step: 100, label: 'draws per chain', format: (v) => String(v) }),
    spread: float(8, { min: 0, max: 20, step: 0.5, label: 'starting spread' }),
    offset: float(0, { min: 0, max: 3, step: 0.05, label: 'last chain centred at δ' }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
  })

  const run = useMemo(() => {
    const m = state.chains
    const noise = Math.sqrt(1 - state.phi ** 2)
    // Each chain has its own random stream, so adding a chain leaves the others unchanged.
    const all = Array.from({ length: m }, (_, c) => {
      const g = stream(state.seed * 1000 + c)
      const mu = c === m - 1 ? state.offset : 0
      let x = (m === 1 ? 1 : -1 + (2 * c) / (m - 1)) * state.spread
      const out: number[] = []
      for (let t = 0; t < state.draws; t++) {
        x = mu + state.phi * (x - mu) + noise * normal(g)
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
  }, [state.chains, state.phi, state.draws, state.spread, state.offset, state.seed])

  const series: SeriesSpec[] = useMemo(() => {
    const many = run.chains.length > 1
    const stride = Math.max(1, Math.ceil(state.draws / TRACE_POINTS))
    return run.chains.map((c, i): SeriesSpec => {
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
  }, [run, state.draws])

  const tau = (1 + state.phi) / (1 - state.phi)

  const xAxis = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis = useAxis({ label: 'x', hold: 'union' })
  return (
    <Figure
      title="Trace plots, split R-hat and effective sample size"
      state={state}
      caption="Several chains whose stationary distribution is N(0, 1), each with its own random stream, moving with autocorrelation φ from starts spread evenly across ± the starting spread. Each chain is a light line; the chains slider sets how many run. The first half of each chain is treated as warm-up, and R-hat and the effective sample size (summed over chains) use every chain shown. With large starting spread, R-hat over all draws is far above 1 until the chains forget their starts. Raise φ: the chains mix slowly, the effective sample size falls towards (draws/2) × chains/τ, and R-hat on the kept half rises. Move the last chain to another mode: every chain on its own looks stationary, but the chains disagree and R-hat flags it."

      readouts={
        <>
          <Readout label="split R-hat, all draws" value={formatNumber(run.rhatAll)} />
          <Readout label="split R-hat, second halves" value={formatNumber(run.rhatKept)} />
          <Readout label="ESS, second halves" value={formatNumber(run.ess)} />
          <Readout label="kept draws" value={run.keptCount} />
          <Readout label="τ in theory" value={formatNumber(tau)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
