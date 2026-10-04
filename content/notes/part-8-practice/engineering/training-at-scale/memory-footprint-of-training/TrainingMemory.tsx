import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const GB = 1e9
const MAX_S = 16384
const LENGTHS = Array.from({ length: 64 }, (_, i) => ((i + 1) * MAX_S) / 64)

type Preset = '1.3' | '7' | '175'
const PRESETS = [
  { value: '1.3', label: '1.3B' },
  { value: '7', label: '7B' },
  { value: '175', label: '175B' },
] as const
/** Width h, heads a and layers L; parameters are about 12 L h². */
const SHAPES: Record<Preset, { h: number; a: number; layers: number }> = {
  '1.3': { h: 2048, a: 16, layers: 24 },
  '7': { h: 4096, a: 32, layers: 32 },
  '175': { h: 12288, a: 96, layers: 96 },
}

type Tp = '1' | '2' | '4' | '8'
const TPS = [
  { value: '1', label: '1' },
  { value: '2', label: '2' },
  { value: '4', label: '4' },
  { value: '8', label: '8' },
] as const

/** Activation bytes for all layers, 16-bit, with tensor and sequence parallelism of degree t (Korthikanti et al.). */
function activations(s: number, b: number, h: number, a: number, layers: number, t: number) {
  const full = (s * b * h * (34 + (5 * a * s) / h) * layers) / t
  const selective = (s * b * h * 34 * layers) / t
  const recompute = (2 * s * b * h * layers) / t
  return { full, selective, recompute }
}

export function TrainingMemory() {
  const state = useFigureState({
    preset: choice<Preset>(PRESETS, '7', { label: 'model' }),
    tp: choice<Tp>(TPS, '1', { label: 'tensor-parallel degree t' }),
    batch: int(1, { min: 1, max: 16, step: 1, label: 'micro-batch size b' }),
    len: int(4096, { min: 256, max: MAX_S, step: 256, label: 'sequence length s' }),
  })
  const { h, a, layers } = SHAPES[state.preset]
  const t = Number(state.tp)
  const modelState = (16 * 12 * layers * h * h) / t

  const series = useMemo(() => {
    const act = LENGTHS.map((s) => activations(s, state.batch, h, a, layers, t))
    return [
      {
        name: 'weights, gradients, Adam (16 bytes each)',
        x: LENGTHS,
        y: LENGTHS.map(() => modelState / GB),
        slot: 0,
      },
      { name: 'activations, stored', x: LENGTHS, y: act.map((v) => v.full / GB), slot: 1 },
      {
        name: 'activations, selective recompute',
        x: LENGTHS,
        y: act.map((v) => v.selective / GB),
        slot: 2,
      },
      { name: 'activations, full recompute', x: LENGTHS, y: act.map((v) => v.recompute / GB), slot: 3 },
      { name: '80 GB device', x: [LENGTHS[0], MAX_S], y: [80, 80], muted: true, dashed: true },
    ] as const
  }, [state.batch, h, a, layers, t, modelState])

  const now = activations(state.len, state.batch, h, a, layers, t)

  const xAxis = useAxis({ label: 'sequence length s (tokens)', range: [LENGTHS[0], MAX_S] })
  const yAxis = useAxis({ label: 'GB per device', hold: 'union', log: true })
  return (
    <Figure
      title="Training memory per device against sequence length"
      purpose="Change the model, the batch size and the tensor-parallel degree to compare model-state and activation memory against sequence length."
      state={state}
      caption="Model state is fixed by the parameter count. Stored activations grow linearly in sequence length s from the 34sbh term and quadratically from the 5as²b attention term. Selective recomputation drops the quadratic term; full recomputation keeps only each layer's input. Tensor parallelism with sequence parallelism divides everything by t. Drag the vertical line to read the values at a length."

      readouts={
        <>
          <Readout label="model state" value={`${formatNumber(modelState / GB)} GB`} />
          <Readout label="activations stored" value={`${formatNumber(now.full / GB)} GB`} />
          <Readout label="selective" value={`${formatNumber(now.selective / GB)} GB`} />
          <Readout label="full recompute" value={`${formatNumber(now.recompute / GB)} GB`} />
          <Readout label="5as/h against 34" value={formatNumber((5 * a * state.len) / h)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Curve {...series[4]} />
        <Handle {...state.handle('len', { label: 's' })} />
      </Plot>
    </Figure>
  )
}
