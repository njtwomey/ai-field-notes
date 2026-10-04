import { useMemo, useState } from 'react'
import { blahutArimotoCapacity } from 'aifn-methods/information/channels'
import { binaryEntropy } from 'aifn/numerics/special'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  Bars,
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Plots,
  Readout,
  Select,
  Slider,
  useAxis,
} from 'aifn-render'

const h2 = (p: number) => binaryEntropy(p, 2)

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

export function ChannelCapacityExplorer() {
  const [channelType, setChannelType] = useState<ChannelName>('Z channel')
  const [noise, setNoise] = useState(0.3)

  const { W, exact } = useMemo(() => channelOf(channelType, noise), [channelType, noise])

  const run = useMemo(
    () =>
      trace(blahutArimotoCapacity(W, { tolerance: 1e-12 }), {}, 200, {
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

  const inputDist = useMemo(
    () => ({
      x: W.map((_, i) => i),
      y: toFlat(last.input),
    }),
    [W, last],
  )

  const itAxis = useAxis({ label: 'iteration k' })
  const bitsAxis = useAxis({ label: 'information (bits)', key: channelType })
  const symAxis = useAxis({
    label: 'input symbol x',
    categories: W.map((_, i) => String(i)),
    range: [-0.5, W.length - 0.5],
  })
  const pxAxis = useAxis({ label: 'p*(x)', range: [0, 1] })

  const capacityBits = last.lower / Math.LN2

  return (
    <Figure
      title="Blahut–Arimoto algorithm: channel capacity"
      purpose="Visualize iterative convergence of Blahut–Arimoto bounds towards channel capacity and the capacity-achieving input distribution across canonical noisy discrete channels."
      caption="Top: lower bound log Σ p exp(D(W(·|x) ‖ q)) and upper bound max_x D(W(·|x) ‖ q) squeezing the capacity C = max_p I(X; Y) until meeting at the optimal mutual information. Bottom: the optimal capacity-achieving input distribution p*(x)."
    >
      <ControlRow>
        <Select
          label="Discrete channel"
          value={channelType}
          onChange={(v) => setChannelType(v as ChannelName)}
          options={[
            { value: 'binary symmetric', label: 'Binary Symmetric Channel (BSC)' },
            { value: 'binary erasure', label: 'Binary Erasure Channel (BEC)' },
            { value: 'Z channel', label: 'Z Channel' },
            { value: 'noisy typewriter (5)', label: 'Noisy Typewriter (K=5)' },
            { value: 'asymmetric 3 × 3', label: 'Asymmetric 3×3 Channel' },
          ]}
        />
        <Slider label="Noise transition error e" value={noise} min={0.01} max={0.99} step={0.01} onChange={setNoise} />
      </ControlRow>

      <div className="my-2 flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
        <Readout label="capacity (bits)" value={formatNumber(capacityBits)} />
        {exact !== undefined && <Readout label="analytical closed form" value={formatNumber(exact)} />}
        <Readout label="iterations" value={last.t} />
        <Readout label="status" value={run.meta.stopped} />
      </div>

      <Plots cols={2}>
        <Plot x={itAxis} y={bitsAxis} title="Upper & lower bounds on capacity">
          <Curve name="Upper bound max_x D" x={bounds.x} y={bounds.upper} slot={0} />
          <Curve name="Lower bound log Σ p exp(D)" x={bounds.x} y={bounds.lower} slot={1} />
          <Curve name="I(p; W) current" x={bounds.x} y={bounds.information} slot={2} dashed />
        </Plot>

        <Plot x={symAxis} y={pxAxis} title="Capacity-achieving input distribution p*(x)">
          <Bars name="p*(x)" x={inputDist.x} y={inputDist.y} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
