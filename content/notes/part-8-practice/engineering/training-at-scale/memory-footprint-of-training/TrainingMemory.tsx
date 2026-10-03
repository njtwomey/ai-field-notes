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
} from 'aifn-render'

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
  const [preset, setPreset] = useState<Preset>('7')
  const [batch, setBatch] = useState(1)
  const [tp, setTp] = useState<Tp>('1')
  const len = useParam(4096, { min: 256, max: MAX_S, step: 256 })
  const { h, a, layers } = SHAPES[preset]
  const t = Number(tp)
  const state = (16 * 12 * layers * h * h) / t

  const series = useMemo((): XYSeries[] => {
    const act = LENGTHS.map((s) => activations(s, batch, h, a, layers, t))
    return [
      {
        name: 'weights, gradients, Adam (16 bytes each)',
        type: 'line',
        x: LENGTHS,
        y: LENGTHS.map(() => state / GB),
        slot: 0,
      },
      { name: 'activations, stored', type: 'line', x: LENGTHS, y: act.map((v) => v.full / GB), slot: 1 },
      {
        name: 'activations, selective recompute',
        type: 'line',
        x: LENGTHS,
        y: act.map((v) => v.selective / GB),
        slot: 2,
      },
      { name: 'activations, full recompute', type: 'line', x: LENGTHS, y: act.map((v) => v.recompute / GB), slot: 3 },
      { name: '80 GB device', type: 'line', x: [LENGTHS[0], MAX_S], y: [80, 80], muted: true, dashed: true },
    ]
  }, [batch, h, a, layers, t, state])

  const now = activations(len.value, batch, h, a, layers, t)
  const handles: Handle[] = [{ kind: 'x', at: len.value, label: 's', onDrag: len.set }]

  return (
    <Interactive
      title="Training memory per device against sequence length"
      caption="Model state is fixed by the parameter count. Stored activations grow linearly in sequence length s from the 34sbh term and quadratically from the 5as²b attention term. Selective recomputation drops the quadratic term; full recomputation keeps only each layer's input. Tensor parallelism with sequence parallelism divides everything by t. Drag the vertical line to read the values at a length."
      controls={
        <>
          <ParamChoice label="model" value={preset} onChange={setPreset} options={PRESETS} />
          <ParamChoice label="tensor-parallel degree t" value={tp} onChange={setTp} options={TPS} />
          <ParamSlider label="micro-batch size b" value={batch} onChange={setBatch} min={1} max={16} step={1} />
          <ParamSlider label="sequence length s" param={len} />
        </>
      }
      readout={
        <>
          <Readout label="model state" value={`${formatNumber(state / GB)} GB`} />
          <Readout label="activations stored" value={`${formatNumber(now.full / GB)} GB`} />
          <Readout label="selective" value={`${formatNumber(now.selective / GB)} GB`} />
          <Readout label="full recompute" value={`${formatNumber(now.recompute / GB)} GB`} />
          <Readout label="5as/h against 34" value={formatNumber((5 * a * len.value) / h)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="sequence length s (tokens)"
        yLabel="GB per device"
        xRange={[LENGTHS[0], MAX_S]}
        yLog
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
