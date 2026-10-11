import { useMemo } from 'react'
import {
  Button,
  choice,
  Figure,
  formatNumber,
  Curve,
  Handle,
  Plot,
  useAxis,
  int,
  Readout,
  useFigureState,
} from 'aifn-render'

const MAX_LEN = 32768
const LENGTHS = Array.from({ length: 65 }, (_, i) => (i * MAX_LEN) / 64)
const GIB = 2 ** 30

type Precision = '4' | '2' | '1' | '0.5'
const PRECISIONS = [
  { value: '4', label: 'fp32' },
  { value: '2', label: 'fp16' },
  { value: '1', label: 'fp8' },
  { value: '0.5', label: '4-bit' },
] as const

type HeadDim = '64' | '128' | '256'
const HEAD_DIMS = [
  { value: '64', label: '64' },
  { value: '128', label: '128' },
  { value: '256', label: '256' },
] as const

/** Cache bytes = 2 (K and V) · layers · KV heads · head width · tokens · batch · bytes per value. */
const cacheGiB = (layers: number, kvHeads: number, dHead: number, bytes: number, batch: number, tokens: number) =>
  (2 * layers * kvHeads * dHead * tokens * batch * bytes) / GIB

export function CacheMemory() {
  const state = useFigureState({
    layers: int(80, { min: 1, max: 128, step: 1, label: 'layers' }),
    heads: int(64, { min: 1, max: 128, step: 1, label: 'query heads' }),
    groups: int(8, { min: 1, max: 64, step: 1, label: 'KV heads for grouped-query' }),
    dHead: choice<HeadDim>(HEAD_DIMS, '128', { label: 'head width' }),
    precision: choice<Precision>(PRECISIONS, '2', { label: 'cache precision' }),
    batch: int(1, { min: 1, max: 64, step: 1, label: 'batch size' }),
    length: int(4096, { min: 0, max: MAX_LEN, step: 256, label: 'tokens cached' }),
  })

  const kvGroups = Math.min(state.groups, state.heads)
  const bytes = Number(state.precision)
  const dh = Number(state.dHead)

  const series = useMemo(() => {
    const line = (name: string, kvHeads: number, slot: number) => ({
      name,
      x: LENGTHS,
      y: LENGTHS.map((n) => cacheGiB(state.layers, kvHeads, dh, bytes, state.batch, n)),
      slot,
    })
    return [
      line(`multi-head (${state.heads} KV heads)`, state.heads, 0),
      line(`grouped-query (${kvGroups} KV heads)`, kvGroups, 1),
      line('multi-query (1 KV head)', 1, 2),
    ]
  }, [state.layers, state.heads, kvGroups, dh, bytes, state.batch])

  const at = (kvHeads: number) => formatNumber(cacheGiB(state.layers, kvHeads, dh, bytes, state.batch, state.length))
  const perToken = (2 * state.layers * kvGroups * dh * bytes) / 1024
  const lengthAxis = useAxis({ label: 'tokens cached per sequence', range: [0, MAX_LEN] })
  const sizeAxis = useAxis({ label: 'cache size (GiB)', range: [0, undefined], hold: 'union' })

  const preset = (l: number, h: number, g: number) => {
    state.set('layers', l)
    state.set('heads', h)
    state.set('groups', g)
    state.set('dHead', '128')
    state.set('precision', '2')
  }

  return (
    <Figure
      title="KV cache memory against sequence length"
      purpose="Change the model's shape, the precision and the batch size to compare KV cache sizes for multi-head, grouped-query and multi-query attention."
      state={state}
      caption="Cache size grows linearly with the number of cached tokens, with a slope set by layers × KV heads × head width × bytes. Grouped-query attention divides the slope by the number of query heads per KV head, and multi-query attention by the number of query heads. Drag the vertical line to read the three sizes at a given length."
      controls={
        <>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => preset(32, 32, 32)}>
              Llama 2 7B
            </Button>
            <Button variant="outline" size="sm" onClick={() => preset(80, 64, 8)}>
              Llama 2 70B
            </Button>
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="multi-head" value={`${at(state.heads)} GiB`} />
          <Readout label="grouped-query" value={`${at(kvGroups)} GiB`} />
          <Readout label="multi-query" value={`${at(1)} GiB`} />
          <Readout label="grouped-query per token" value={`${formatNumber(perToken)} KiB`} />
        </>
      }
    >
      <Plot x={lengthAxis} y={sizeAxis} height={320}>
        {series.map((line) => (
          <Curve key={line.slot} {...line} />
        ))}
        <Handle {...state.handle('length', { label: 'tokens' })} />
      </Plot>
    </Figure>
  )
}
