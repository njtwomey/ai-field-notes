import { autocorrelation, effectiveSampleSize, monteCarloStandardError, splitRhat } from 'aifn/mcmc'
import { histogram, quantile, runningMean } from 'aifn/stats'
import { linspace, toFlat, type Tensor } from 'aifn/tensor'
import { useMemo, useState } from 'react'
import { Select } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'

export type ChainViewProps = FrameProps & {
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

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))

/**
 * MCMC output for one parameter at a time: trace plots of every chain, a histogram of the pooled draws (against the
 * target's marginal when given), each chain's autocorrelation function, and each chain's running mean. Readouts give
 * the bulk and tail ESS, the rank-normalised split R̂ and the MCSE of the mean, all from `aifn/mcmc`.
 */
export function ChainView({
  draws,
  names,
  steps,
  density,
  means,
  maxLag,
  maxChains = 8,
  title = 'Chains, their histogram, autocorrelation and running means',
  controls,
  readouts,
  ...frame
}: ChainViewProps) {
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
    const trace: XYSeries[] = shown.map((c, j) => ({
      name: `chain ${j}`,
      type: 'line',
      x: xs,
      y: c,
      slot: j,
      thin: m > 1,
    }))
    const pooled = chains.flat()
    const lo = quantile(pooled, 0.002)
    const hi = quantile(pooled, 0.998)
    const h = histogram(pooled, { bins: 50, range: [lo, hi] })
    const scale = (pooled.length - h.dropped) / pooled.length
    const hist: XYSeries[] = [
      {
        name: 'draws',
        type: 'bar',
        thin: true,
        x: Array.from(h.counts, (_, i) => (h.edges[i] + h.edges[i + 1]) / 2),
        y: Array.from(h.density, (v) => v * scale),
      },
    ]
    if (density) {
      const grid = toFlat(linspace(lo, hi, 201))
      hist.push({ name: 'target', type: 'line', x: grid, y: grid.map((x) => density(x, param)) })
    }
    const lags = Math.min(maxLag ?? 60, n - 1)
    const acf: XYSeries[] = shown.map((c, j) => ({
      name: `chain ${j}`,
      type: 'line',
      x: Array.from({ length: lags + 1 }, (_, i) => i),
      y: toFlat(autocorrelation(c, { maxLag: lags })),
      slot: j,
      thin: m > 1,
    }))
    const running: XYSeries[] = shown.map((c, j) => ({
      name: `chain ${j}`,
      type: 'line',
      x: xs,
      y: toFlat(runningMean(c)),
      slot: j,
      thin: m > 1,
    }))
    if (means?.[param] !== undefined)
      running.push({
        name: 'true mean',
        type: 'line',
        x: [xs[0], xs[xs.length - 1]],
        y: [means[param], means[param]],
        emphasis: true,
        dashed: true,
      })
    const diag = {
      bulk: effectiveSampleSize(chains),
      tail: effectiveSampleSize(chains, { method: 'tail' }),
      rhat: splitRhat(chains),
      mcse: monteCarloStandardError(chains),
    }
    return { trace, hist, acf, running, diag }
  }, [chains, steps, n, m, maxChains, density, param, maxLag, means])

  const options = Array.from({ length: d }, (_, i) => ({ value: String(i), label: names?.[i] ?? `x${i}` }))
  return (
    <Figure
      title={title}
      defaultSize="L"
      {...frame}
      controls={
        <>
          {controls}
          {d > 1 && (
            <Select label="parameter" value={String(param)} onChange={(v) => setK(Number(v))} options={options} />
          )}
        </>
      }
      readouts={
        <>
          {readouts}
          <Readout label={`${m} × ${n} draws; bulk ESS`} value={f3(view.diag.bulk)} />
          <Readout label="tail ESS" value={f3(view.diag.tail)} />
          <Readout label="R̂ (rank-normalised split)" value={f3(view.diag.rhat)} />
          <Readout label="MCSE of the mean" value={f3(view.diag.mcse)} />
        </>
      }
    >
      <Subplots rows={2} cols={2} heightRatios={[1, 1]}>
        <Panel>
          <XYChart series={view.trace} xLabel="step" yLabel={label} legend={false} />
        </Panel>
        <Panel>
          <XYChart series={view.hist} xLabel={label} yLabel="density" />
        </Panel>
        <Panel>
          <XYChart series={view.acf} xLabel="lag" yLabel="autocorrelation" legend={false} />
        </Panel>
        <Panel>
          <XYChart series={view.running} xLabel="step" yLabel={`running mean of ${label}`} legend={false} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
