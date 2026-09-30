import * as Channels from 'aifn-applied/information/channels'
import { binaryEntropy } from 'aifn/numerics/special'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Readout, XYChart, type XYSeries } from '@lab/viz'
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
  const [name, setName] = useState<ChannelName>('Z channel')
  const [e, setE] = useState(0.3)
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
  const series = useMemo((): XYSeries[] => {
    const x = run.index
    return [
      { name: 'upper bound maxₓ D(W‖q)', type: 'line', x, y: toFlat(run.series.upper), slot: 0 },
      { name: 'lower bound log Σ p e^D', type: 'line', x, y: toFlat(run.series.lower), slot: 1 },
      { name: 'I(p; W)', type: 'line', x, y: toFlat(run.series.information), slot: 2, dashed: true },
    ]
  }, [run])
  const input = useMemo(
    (): XYSeries[] => [
      { name: 'capacity-achieving p(x)', type: 'bar', x: W.map((_, i) => i), y: toFlat(last.input), slot: 0 },
    ],
    [W, last],
  )
  return (
    <Figure
      title="Blahut–Arimoto: channel capacity"
      defaultSize="L"
      controls={
        <>
          <Select
            label="channel"
            value={name}
            onChange={setName}
            options={['binary symmetric', 'binary erasure', 'Z channel', 'noisy typewriter (5)', 'asymmetric 3 × 3']}
          />
          <Slider label="noise e" value={e} min={0} max={0.99} onChange={setE} />
        </>
      }
      readouts={
        <>
          <Readout label="capacity (bits)" value={formatValue(last.lower / Math.LN2)} />
          {exact !== undefined && <Readout label="closed form" value={formatValue(exact)} />}
          <Readout label="iterations" value={last.t} />
          <Readout label="stopped" value={run.meta.stopped} />
        </>
      }
      caption="Each step reweights the input by exp D(W(·|x) ‖ q); the bounds squeeze the capacity from both sides and the run stops when they meet within 10⁻¹² nats."
    >
      <div className="flex flex-col gap-4">
        <ChartSize scale={0.6}>
          <XYChart series={series} xLabel="iteration" yLabel="bits" integerX rescaleOnChange={false} holdFit="union" />
        </ChartSize>
        <ChartSize scale={0.4}>
          <XYChart series={input} xLabel="input symbol x" yLabel="p(x)" integerX yRange={[0, 1]} />
        </ChartSize>
      </div>
    </Figure>
  )
}

// ── Rate–distortion ──────────────────────────────────────────────────────────────────────────────────────────────────

export function RateDistortionSpecimen() {
  const [p, setP] = useState(0.3)
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
  const series = useMemo((): XYSeries[] => {
    const ds = Array.from({ length: 200 }, (_, i) => (Math.min(p, 1 - p) * i) / 199)
    return [
      { name: 'Blahut–Arimoto points', type: 'scatter', x: toFlat(curve.distortion), y: toFlat(curve.rate), slot: 0 },
      { name: 'H(p) − H(D)', type: 'line', x: ds, y: ds.map((d) => h2(p) - h2(d)), slot: 1, dashed: true },
    ]
  }, [curve, p])
  return (
    <Figure
      title="Rate–distortion of a binary source under Hamming distortion"
      controls={<Slider label="source p(1)" value={p} min={0.01} max={0.5} onChange={setP} />}
      readouts={
        <>
          <Readout label="R(0) = H(p) (bits)" value={formatValue(h2(p))} />
          <Readout label="all converged" value={String(curve.converged)} />
        </>
      }
      caption="Each point is one Blahut–Arimoto run at slope −β; together they trace R(D) = H(p) − H(D) for D ≤ min(p, 1 − p)."
    >
      <XYChart
        series={series}
        xLabel="distortion D"
        yLabel="rate R (bits)"
        yRange={[0, undefined]}
        rescaleOnChange={false}
        holdFit="union"
      />
    </Figure>
  )
}
