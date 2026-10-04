import { useMemo } from 'react'
import {
  choice,
  Figure,
  int,
  formatNumber,
  Curve,
  Handle,
  Plot,
  Plots,
  useAxis,
  Readout,
  useFigureState,
} from 'aifn-render'

// NVIDIA A100 80 GB SXM: 2.039 TB/s memory bandwidth, 312 TFLOP/s dense bf16, 80 GB of memory.
const BANDWIDTH = 2.039e12
const PEAK = 312e12
const CAPACITY = 80e9
// Llama 2 7B shape: 6.7e9 parameters, 32 layers, 32 query heads of width 128.
const PARAMS = 6.7e9
const LAYERS = 32
const HEAD_DIM = 128
const MAX_BATCH = 256
const BATCHES = Array.from({ length: MAX_BATCH }, (_, i) => i + 1)

type Bits = '16' | '8' | '4'
const BITS = [
  { value: '16', label: '16-bit' },
  { value: '8', label: '8-bit' },
  { value: '4', label: '4-bit' },
] as const

type KvHeads = '32' | '8'
const KV_HEADS = [
  { value: '32', label: 'multi-head (32)' },
  { value: '8', label: 'grouped-query (8)' },
] as const

/** Time of one decode step for a batch: the larger of memory traffic time and arithmetic time. */
function step(batch: number, weightBytes: number, kvPerToken: number, context: number) {
  const memory = (weightBytes + batch * context * kvPerToken) / BANDWIDTH
  const compute = (2 * PARAMS * batch) / PEAK
  return { time: Math.max(memory, compute), memoryBound: memory >= compute }
}

export function ServingRoofline() {
  const state = useFigureState({
    batch: int(16, { min: 1, max: MAX_BATCH, label: 'batch size' }),
    context: int(1024, { min: 128, max: 8192, step: 128, label: 'tokens of context per sequence' }),
    bits: choice<Bits>(BITS, '16', { label: 'weight precision' }),
    kvHeads: choice<KvHeads>(KV_HEADS, '32', { label: 'KV heads' }),
  })

  const weightBytes = (PARAMS * Number(state.bits)) / 8
  const kvPerToken = 2 * LAYERS * Number(state.kvHeads) * HEAD_DIM * 2
  const maxBatch = Math.floor((CAPACITY - weightBytes) / (state.context * kvPerToken))

  const { throughput, latency } = useMemo(() => {
    const t = BATCHES.map((b) => step(b, weightBytes, kvPerToken, state.context).time)
    const fits = (b: number) => b <= maxBatch
    const split = (name: string, y: number[], slot: number) => [
      { name, type: 'line', x: BATCHES.filter(fits), y: y.filter((_, i) => fits(BATCHES[i])), slot },
      {
        name: 'does not fit in 80 GB',
        type: 'line',
        x: BATCHES.filter((b) => !fits(b)),
        y: y.filter((_, i) => !fits(BATCHES[i])),
        muted: true,
      },
    ]
    return {
      throughput: split(
        'tokens per second',
        t.map((s, i) => BATCHES[i] / s),
        0,
      ),
      latency: split(
        'ms per token',
        t.map((s) => 1000 * s),
        1,
      ),
    }
  }, [weightBytes, kvPerToken, state.context, maxBatch])

  const now = step(state.batch, weightBytes, kvPerToken, state.context)
  const batchAxis = useAxis({ label: 'batch size', range: [1, MAX_BATCH] })
  const throughputAxis = useAxis({ label: 'tokens per second', range: [0, undefined], hold: 'union' })
  const latencyAxis = useAxis({ label: 'ms per token', range: [0, undefined], hold: 'union' })

  return (
    <Figure
      title="Decode throughput and latency of a 7B model on one A100"
      purpose="Change the batch size, context, weight precision and KV heads to see decode throughput and latency and what limits them."
      state={state}
      caption="Each decode step reads every weight and every cached key and value once, and does two operations per weight per sequence. At small batch sizes the weights dominate the traffic, so throughput grows almost linearly with batch size at little cost in latency. At larger batches the KV cache dominates, and throughput levels off. Grey marks batch sizes whose cache does not fit in memory. Drag either vertical line to change the batch size."

      readouts={
        <>
          <Readout label="step time" value={`${formatNumber(1000 * now.time)} ms`} />
          <Readout label="throughput" value={`${formatNumber(state.batch / now.time)} tokens/s`} />
          <Readout label="limited by" value={now.memoryBound ? 'memory bandwidth' : 'arithmetic'} />
          <Readout label="largest batch that fits" value={formatNumber(Math.max(maxBatch, 0))} />
        </>
      }
    >
      <Plots rows={2}>
        <Plot x={batchAxis} y={throughputAxis}>
          <Curve {...throughput[0]} />
          <Curve {...throughput[1]} />
          <Handle {...state.handle('batch', { label: 'batch' })} />
        </Plot>
        <Plot x={batchAxis} y={latencyAxis}>
          <Curve {...latency[0]} />
          <Curve {...latency[1]} />
          <Handle {...state.handle('batch', { label: 'batch' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
