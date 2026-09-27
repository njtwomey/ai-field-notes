import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'

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
  const batch = useParam(16, { min: 1, max: MAX_BATCH, step: 1 })
  const [context, setContext] = useState(1024)
  const [bits, setBits] = useState<Bits>('16')
  const [kvHeads, setKvHeads] = useState<KvHeads>('32')

  const weightBytes = (PARAMS * Number(bits)) / 8
  const kvPerToken = 2 * LAYERS * Number(kvHeads) * HEAD_DIM * 2
  const maxBatch = Math.floor((CAPACITY - weightBytes) / (context * kvPerToken))

  const { throughput, latency } = useMemo(() => {
    const t = BATCHES.map((b) => step(b, weightBytes, kvPerToken, context).time)
    const fits = (b: number) => b <= maxBatch
    const split = (name: string, y: number[], slot: number): XYSeries[] => [
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
  }, [weightBytes, kvPerToken, context, maxBatch])

  const now = step(batch.value, weightBytes, kvPerToken, context)
  const handles: Handle[] = [{ kind: 'x', at: batch.value, label: 'batch', onDrag: batch.set }]

  return (
    <Interactive
      title="Decode throughput and latency of a 7B model on one A100"
      caption="Each decode step reads every weight and every cached key and value once, and does two operations per weight per sequence. At small batch sizes the weights dominate the traffic, so throughput grows almost linearly with batch size at little cost in latency. At larger batches the KV cache dominates, and throughput levels off. Grey marks batch sizes whose cache does not fit in memory. Drag either vertical line to change the batch size."
      controls={
        <>
          <ParamSlider label="batch size" param={batch} />
          <ParamSlider
            label="tokens of context per sequence"
            value={context}
            onChange={setContext}
            min={128}
            max={8192}
            step={128}
          />
          <ParamChoice label="weight precision" value={bits} onChange={setBits} options={BITS} />
          <ParamChoice label="KV heads" value={kvHeads} onChange={setKvHeads} options={KV_HEADS} />
        </>
      }
      readout={
        <>
          <Readout label="step time" value={`${formatNumber(1000 * now.time)} ms`} />
          <Readout label="throughput" value={`${formatNumber(batch.value / now.time)} tokens/s`} />
          <Readout label="limited by" value={now.memoryBound ? 'memory bandwidth' : 'arithmetic'} />
          <Readout label="largest batch that fits" value={formatNumber(Math.max(maxBatch, 0))} />
        </>
      }
    >
      <XYChart
        series={throughput}
        xLabel="batch size"
        yLabel="tokens per second"
        xRange={[1, MAX_BATCH]}
        yRange={[0, undefined]}
        handles={handles}
        height={220}
      />
      <XYChart
        series={latency}
        xLabel="batch size"
        yLabel="ms per token"
        xRange={[1, MAX_BATCH]}
        yRange={[0, undefined]}
        handles={handles}
        height={220}
      />
    </Interactive>
  )
}
