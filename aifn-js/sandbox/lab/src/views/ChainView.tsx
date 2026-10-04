import { effectiveSampleSize, monteCarloStandardError, splitRhat } from 'aifn/inference/stochastic'
import { autocorrelation, histogram, quantile, runningMean } from 'aifn/probability/stats'
import { linspace, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Select } from '@lab/controls'
import { PanelSlot } from '@lab/layout'
import { Bars, Curve, Plot, Plots, Readout, useAxis } from '@lab/viz'
import { formatValue } from './format'
import { histogramBars } from './histogram'
import { registerView } from './registry'

export type ChainPanelProps = {
  /** MCMC draws: m chains × n draws × d parameters (as `sampleChains` returns), or m × n for one parameter. */
  draws: Tensor
  /** Parameter names (default x₀, x₁, …). */
  names?: readonly string[]
  /** Step number of each draw (default 1 … n). */
  steps?: readonly number[]
  /** The target's marginal density of parameter k, drawn over the histogram. */
  density?: (x: number, k: number) => number
  /** True marginal means, drawn on the running-mean panel. */
  means?: readonly number[]
  /** Largest ACF lag. Default min(n − 1, 60). */
  maxLag?: number
  /** Most chains drawn in the trace and running-mean panels. Default 8. */
  maxChains?: number
}

const CHAIN_TITLE = 'Chains, their histogram, autocorrelation and running means'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))

/**
 * MCMC output for one parameter at a time: trace plots of every chain, a histogram of the pooled draws (against the
 * target's marginal when given), each chain's autocorrelation function, and each chain's running mean. Readouts give
 * the bulk and tail ESS, the rank-normalised split R̂ and the MCSE of the mean, all from `aifn/inference/stochastic`.
 */
export function ChainPanel({ draws, names, steps, density, means, maxLag, maxChains = 8 }: ChainPanelProps) {
  const [m, n, d = 1] = draws.shape
  const [k, setK] = useState(0)
  const param = Math.min(k, d - 1)
  const label = names?.[param] ?? `x${param}`

  const chains = useMemo(() => {
    const flat = toFlat(draws)
    return Array.from({ length: m }, (_, j) => Array.from({ length: n }, (_, i) => flat[(j * n + i) * d + param]))
  }, [draws, m, n, d, param])

  const view = useMemo(() => {
    const xs = steps ? [...steps] : Array.from({ length: n }, (_, i) => i + 1)
    const shown = chains.slice(0, maxChains)
    const pooled = chains.flat()
    const lo = quantile(pooled, 0.002)
    const hi = quantile(pooled, 0.998)
    const h = histogram(pooled, { bins: 50, range: [lo, hi] })
    const bars = histogramBars(h)
    const scale = (pooled.length - h.dropped) / pooled.length
    const grid = density ? toFlat(linspace(lo, hi, 201)) : null
    const lags = Math.min(maxLag ?? 60, n - 1)
    const lagX = Array.from({ length: lags + 1 }, (_, i) => i)
    const diag = {
      bulk: effectiveSampleSize(chains),
      tail: effectiveSampleSize(chains, { method: 'tail' }),
      rhat: splitRhat(chains),
      mcse: monteCarloStandardError(chains),
    }
    return {
      xs,
      shown,
      bars: { edges: bars.edges, x: bars.x, y: bars.density.map((v) => v * scale) },
      target: grid ? { x: grid, y: grid.map((x) => density!(x, param)) } : null,
      acf: shown.map((c) => ({ x: lagX, y: toFlat(autocorrelation(c, { maxLag: lags })) })),
      running: shown.map((c) => toFlat(runningMean(c))),
      diag,
    }
  }, [chains, steps, n, maxChains, density, param, maxLag])

  const thin = m > 1
  const stepAxis = useAxis({ label: 'step' })
  const valueAxis = useAxis({ label })
  const densityValue = useAxis({ label })
  const densityAxis = useAxis({ label: 'density' })
  const lagAxis = useAxis({ label: 'lag' })
  const acfAxis = useAxis({ label: 'autocorrelation' })
  const runStepAxis = useAxis({ label: 'step' })
  const runAxis = useAxis({ label: `running mean of ${label}` })
  const ends = [view.xs[0], view.xs[view.xs.length - 1]]
  const trueMean = means?.[param]
  const options = Array.from({ length: d }, (_, i) => ({ value: String(i), label: names?.[i] ?? `x${i}` }))
  return (
    <>
      {d > 1 && (
        <PanelSlot slot="controls">
          <Select label="parameter" value={String(param)} onChange={(v) => setK(Number(v))} options={options} />
        </PanelSlot>
      )}
      <PanelSlot slot="readouts">
        <>
          <Readout label={`${m} × ${n} draws; bulk ESS`} value={f3(view.diag.bulk)} />
          <Readout label="tail ESS" value={f3(view.diag.tail)} />
          <Readout label="R̂ (rank-normalised split)" value={f3(view.diag.rhat)} />
          <Readout label="MCSE of the mean" value={f3(view.diag.mcse)} />
        </>
      </PanelSlot>
      <Plots rows={2} cols={2}>
        <Plot x={stepAxis} y={valueAxis} legend={false}>
          {view.shown.map((c, j) => (
            <Curve key={j} name={`chain ${j}`} x={view.xs} y={c} slot={j} thin={thin} />
          ))}
        </Plot>
        <Plot x={densityValue} y={densityAxis}>
          <Bars name="draws" x={view.bars.x} y={view.bars.y} edges={view.bars.edges} />
          {view.target && <Curve name="target" x={view.target.x} y={view.target.y} />}
        </Plot>
        <Plot x={lagAxis} y={acfAxis} legend={false}>
          {view.acf.map((c, j) => (
            <Curve key={j} name={`chain ${j}`} x={c.x} y={c.y} slot={j} thin={thin} />
          ))}
        </Plot>
        <Plot x={runStepAxis} y={runAxis} legend={false}>
          {view.running.map((y, j) => (
            <Curve key={j} name={`chain ${j}`} x={view.xs} y={y} slot={j} thin={thin} />
          ))}
          {trueMean !== undefined && <Curve name="true mean" x={ends} y={[trueMean, trueMean]} emphasis dashed />}
        </Plot>
      </Plots>
    </>
  )
}

// MCMC draws are a plain tensor (m × n × d); the view is drawn when asked for by key ('chains/diagnostics').
registerView<Tensor>({
  key: 'chains/diagnostics',
  kind: 'chains',
  description: 'MCMC chains: traces, pooled histogram, autocorrelation and running means, with ESS, split R̂ and MCSE.',
  title: () => CHAIN_TITLE,
  render: (draws) => <ChainPanel draws={draws} />,
})
