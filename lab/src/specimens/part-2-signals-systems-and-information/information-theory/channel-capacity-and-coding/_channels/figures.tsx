import * as Channels from 'aifn-methods/information/channels'
import { binaryEntropy } from 'aifn-compute/numerics/special'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { useMemo } from 'react'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, useFigureState } from 'aifn-render/state'
import { Bars, Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const h2 = (p: number) => binaryEntropy(p, 2)

// ── Blahut–Arimoto capacity ──────────────────────────────────────────────────────────────────────────────────────────

type ChannelName = 'binary symmetric' | 'binary erasure' | 'Z channel' | 'noisy typewriter (5)' | 'asymmetric 3 × 3'

function channelOf(name: ChannelName, e: number): { W: number[][]; exact?: number } {
  switch (name) {
    case 'binary symmetric':
      return {
        W: [
          [1 - e, e],
          [e, 1 - e],
        ],
        exact: 1 - h2(e),
      }
    case 'binary erasure':
      return {
        W: [
          [1 - e, e, 0],
          [0, e, 1 - e],
        ],
        exact: 1 - e,
      }
    case 'Z channel':
      // Capacity log₂(1 + (1 − e) e^{e/(1−e)}) for a 1 → 0 flip with probability e.
      return {
        W: [
          [1, 0],
          [e, 1 - e],
        ],
        exact: e < 1 ? Math.log2(1 + (1 - e) * e ** (e / (1 - e))) : 0,
      }
    case 'noisy typewriter (5)':
      return {
        W: Array.from({ length: 5 }, (_, i) =>
          Array.from({ length: 5 }, (_, j) => (j === i ? 1 - e : j === (i + 1) % 5 ? e : 0)),
        ),
      }
    case 'asymmetric 3 × 3':
      return {
        W: [
          [1 - e, e / 2, e / 2],
          [e / 4, 1 - e / 2, e / 4],
          [0.1, 0.3, 0.6],
        ],
      }
  }
}

export function CapacitySpecimen() {
  const state = useFigureState({
    channel: row('channel', {
      name: choice(
        [
          'binary symmetric',
          'binary erasure',
          'Z channel',
          'noisy typewriter (5)',
          'asymmetric 3 × 3',
        ] as ChannelName[],
        'Z channel',
        { label: 'channel' },
      ),
      e: slider(0, 0.99, 0.3, { label: 'noise e' }),
    }),
  })
  const name = state.channel.name as ChannelName
  const { e } = state.channel
  const { W, exact } = useMemo(() => channelOf(name, e), [name, e])
  const run = useMemo(
    () =>
      trace(Channels.blahutArimotoCapacity(W, { tolerance: 1e-12 }), {}, 200, {
        record: {
          lower: (s) => s.lower / Math.LN2,
          upper: (s) => s.upper / Math.LN2,
          information: (s) => s.information / Math.LN2,
        },
      }),
    [W],
  )
  const last = run.steps[run.steps.length - 1]
  const bounds = useMemo(
    () => ({
      x: Array.from(run.index),
      upper: toFlat(run.series.upper),
      lower: toFlat(run.series.lower),
      information: toFlat(run.series.information),
    }),
    [run],
  )
  const input = useMemo(() => ({ x: W.map((_, i) => i), y: toFlat(last.input) }), [W, last])
  const it = useAxis({ label: 'iteration', hold: 'union' })
  const bits = useAxis({ label: 'bits', hold: 'union', key: name })
  const sym = useAxis({ label: 'input symbol x', categories: W.map((_, i) => String(i)) })
  const px = useAxis({ label: 'p(x)', range: [0, 1] })
  return (
    <Figure
      purpose="Blahut–Arimoto finds a channel's capacity by alternating between the input distribution and the output it induces; an upper and a lower bound squeeze the capacity from both sides."
      title="Blahut–Arimoto: channel capacity"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="capacity (bits)" value={formatValue(last.lower / Math.LN2)} />
          {exact !== undefined && <Readout label="closed form" value={formatValue(exact)} />}
          <Readout label="iterations" value={last.t} />
          <Readout label="stopped" value={run.meta.stopped} />
        </>
      }
      caption="Top: the bounds per iteration; bottom: the capacity-achieving input distribution. Each step reweights the input by exp D(W(·|x) ‖ q); the bounds squeeze the capacity from both sides and the run stops when they meet within 10⁻¹² nats."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={it} y={bits}>
          <Curve name="upper bound maxₓ D(W‖q)" x={bounds.x} y={bounds.upper} slot={0} />
          <Curve name="lower bound log Σ p e^D" x={bounds.x} y={bounds.lower} slot={1} />
          <Curve name="I(p; W)" x={bounds.x} y={bounds.information} slot={2} dashed />
        </Plot>
        <Plot x={sym} y={px} legend={false}>
          <Bars name="capacity-achieving p(x)" x={input.x} y={input.y} slot={0} width={0.6} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Rate–distortion ──────────────────────────────────────────────────────────────────────────────────────────────────

export function RateDistortionSpecimen() {
  const state = useFigureState({ p: slider(0.01, 0.5, 0.3, { label: 'source p(1)' }) })
  const { p } = state
  const curve = useMemo(() => {
    const betas = Array.from({ length: 60 }, (_, i) => 0.05 * 1.12 ** i)
    return Channels.rateDistortionCurve(
      [1 - p, p],
      [
        [0, 1],
        [1, 0],
      ],
      betas,
      { base: 2 },
    )
  }, [p])
  const closed = useMemo(() => {
    const ds = Array.from({ length: 200 }, (_, i) => (Math.min(p, 1 - p) * i) / 199)
    return { x: ds, y: ds.map((d) => h2(p) - h2(d)) }
  }, [p])
  const points = useMemo(() => ({ x: toFlat(curve.distortion), y: toFlat(curve.rate) }), [curve])
  const da = useAxis({ label: 'distortion D', hold: 'union' })
  const ra = useAxis({ label: 'rate R (bits)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      purpose="Allowing a fraction D of bits to be wrong lowers the rate needed to describe a binary source from H(p) to H(p) − H(D); Blahut–Arimoto traces the curve point by point."
      title="Rate–distortion of a binary source under Hamming distortion"
      state={state}
      readouts={
        <>
          <Readout label="R(0) = H(p) (bits)" value={formatValue(h2(p))} />
          <Readout label="all converged" value={String(curve.converged)} />
        </>
      }
      caption="Each point is one Blahut–Arimoto run at slope −β; together they trace R(D) = H(p) − H(D) for D ≤ min(p, 1 − p)."
    >
      <Plot x={da} y={ra}>
        <Points name="Blahut–Arimoto points" x={points.x} y={points.y} slot={0} />
        <Curve name="H(p) − H(D)" x={closed.x} y={closed.y} slot={1} dashed />
      </Plot>
    </Figure>
  )
}
