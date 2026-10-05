import { useMemo } from 'react'
import { PanelSlot } from 'aifn-render/layout'
import { Curve, Plot, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from './format'

export type ElboPanelProps = {
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
export function ElboPanel({ elbo, steps, logEvidence, extra, at }: ElboPanelProps) {
  const { x, y } = useMemo(() => {
    const y = Array.from(elbo)
    return { y, x: steps ? Array.from(steps) : y.map((_, i) => i) }
  }, [elbo, steps])
  const current =
    at === undefined
      ? -1
      : Math.max(
          0,
          x.findIndex((v) => v >= at),
        )
  const xAxis = useAxis({ label: 'step' })
  const yAxis = useAxis({ label: 'ELBO' })
  const last = elbo[elbo.length - 1]
  return (
    <>
      <PanelSlot slot="readouts">
        <>
          <Readout label="final ELBO" value={f4(last)} />
          {logEvidence !== undefined && <Readout label="log p(x)" value={f4(logEvidence)} />}
          {logEvidence !== undefined && <Readout label="KL(q ‖ p) = log p(x) − ELBO" value={f4(logEvidence - last)} />}
        </>
      </PanelSlot>
      <Plot x={xAxis} y={yAxis}>
        <Curve name="ELBO" x={x} y={y} slot={0} thin={!!extra?.length} />
        {extra?.map((e, i) => (
          <Curve key={e.name} name={e.name} x={x} y={e.y} slot={i + 1} />
        ))}
        {logEvidence !== undefined && (
          <Curve name="log p(x)" x={[x[0], x[x.length - 1]]} y={[logEvidence, logEvidence]} emphasis dashed />
        )}
        {current >= 0 && <Points name="current" x={[x[current]]} y={[y[current]]} emphasis live />}
      </Plot>
    </>
  )
}
