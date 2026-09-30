import { useMemo } from 'react'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type XYSeries } from '@lab/viz'
import { formatValue } from './format'
import type { FrameProps } from './frame'

export type ElboViewProps = FrameProps & {
  /** The ELBO at each recorded step (noisy for stochastic VI). */
  elbo: ArrayLike<number>
  /** Step number of each value (default 0, 1, …). */
  steps?: ArrayLike<number>
  /** log p(x) when it is known: the ELBO's ceiling, and the gap is KL(q ‖ p). */
  logEvidence?: number
  /** Extra traces over the same steps, e.g. a low-noise ELBO estimate beside the optimiser's noisy one. */
  extra?: readonly { name: string; y: ArrayLike<number> }[]
  /** A marker at this step, e.g. a player's position. */
  at?: number
}

const f4 = (v: number) => formatValue(Number(v.toPrecision(4)))

/**
 * An ELBO trace: the lower bound against the step, with log p(x) as its ceiling when known, so the gap is the reverse
 * divergence KL(q ‖ p). Coordinate ascent rises monotonically; stochastic VI is noisy, so pass a low-noise estimate in `extra`.
 */
export function ElboView({
  elbo,
  steps,
  logEvidence,
  extra,
  at,
  title = 'ELBO against the step',
  readouts,
  ...frame
}: ElboViewProps) {
  const series = useMemo(() => {
    const y = Array.from(elbo)
    const x = steps ? Array.from(steps) : y.map((_, i) => i)
    const out: XYSeries[] = [{ name: 'ELBO', type: 'line', x, y, slot: 0, thin: !!extra?.length }]
    extra?.forEach((e, i) => out.push({ name: e.name, type: 'line', x, y: Array.from(e.y), slot: i + 1 }))
    if (logEvidence !== undefined)
      out.push({
        name: 'log p(x)',
        type: 'line',
        x: [x[0], x[x.length - 1]],
        y: [logEvidence, logEvidence],
        emphasis: true,
        dashed: true,
      })
    if (at !== undefined) {
      const i = Math.max(
        0,
        x.findIndex((v) => v >= at),
      )
      out.push({ name: 'current', type: 'scatter', x: [x[i]], y: [y[i]], emphasis: true })
    }
    return out
  }, [elbo, steps, logEvidence, extra, at])
  const last = elbo[elbo.length - 1]
  return (
    <Figure
      title={title}
      {...frame}
      readouts={
        <>
          {readouts}
          <Readout label="final ELBO" value={f4(last)} />
          {logEvidence !== undefined && <Readout label="log p(x)" value={f4(logEvidence)} />}
          {logEvidence !== undefined && <Readout label="KL(q ‖ p) = log p(x) − ELBO" value={f4(logEvidence - last)} />}
        </>
      }
    >
      <XYChart series={series} xLabel="step" yLabel="ELBO" />
    </Figure>
  )
}
