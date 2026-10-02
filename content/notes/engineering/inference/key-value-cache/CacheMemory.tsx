import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
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
  const [layers, setLayers] = useState(80)
  const [heads, setHeads] = useState(64)
  const [groups, setGroups] = useState(8)
  const [dHead, setDHead] = useState<HeadDim>('128')
  const [precision, setPrecision] = useState<Precision>('2')
  const [batch, setBatch] = useState(1)
  const length = useParam(4096, { min: 0, max: MAX_LEN, step: 256 })

  const kvGroups = Math.min(groups, heads)
  const bytes = Number(precision)
  const dh = Number(dHead)

  const series = useMemo((): XYSeries[] => {
    const line = (name: string, kvHeads: number, slot: number): XYSeries => ({
      name,
      type: 'line',
      x: LENGTHS,
      y: LENGTHS.map((n) => cacheGiB(layers, kvHeads, dh, bytes, batch, n)),
      slot,
    })
    return [
      line(`multi-head (${heads} KV heads)`, heads, 0),
      line(`grouped-query (${kvGroups} KV heads)`, kvGroups, 1),
      line('multi-query (1 KV head)', 1, 2),
    ]
  }, [layers, heads, kvGroups, dh, bytes, batch])

  const at = (kvHeads: number) => formatNumber(cacheGiB(layers, kvHeads, dh, bytes, batch, length.value))
  const perToken = (2 * layers * kvGroups * dh * bytes) / 1024
  const handles: Handle[] = [{ kind: 'x', at: length.value, label: 'tokens', onDrag: length.set }]

  const preset = (l: number, h: number, g: number) => {
    setLayers(l)
    setHeads(h)
    setGroups(g)
    setDHead('128')
    setPrecision('2')
  }

  return (
    <Interactive
      title="KV cache memory against sequence length"
      caption="Cache size grows linearly with the number of cached tokens, with a slope set by layers × KV heads × head width × bytes. Grouped-query attention divides the slope by the number of query heads per KV head, and multi-query attention by the number of query heads. Drag the vertical line to read the three sizes at a given length."
      controls={
        <>
          <ParamSlider label="layers" value={layers} onChange={setLayers} min={1} max={128} step={1} />
          <ParamSlider label="query heads" value={heads} onChange={setHeads} min={1} max={128} step={1} />
          <ParamSlider
            label="KV heads for grouped-query"
            value={groups}
            onChange={setGroups}
            min={1}
            max={64}
            step={1}
          />
          <ParamChoice label="head width" value={dHead} onChange={setDHead} options={HEAD_DIMS} />
          <ParamChoice label="cache precision" value={precision} onChange={setPrecision} options={PRECISIONS} />
          <ParamSlider label="batch size" value={batch} onChange={setBatch} min={1} max={64} step={1} />
          <ParamSlider label="tokens cached" param={length} />
          <div className="flex flex-wrap gap-2">
            <ParamButton onClick={() => preset(32, 32, 32)}>Llama 2 7B</ParamButton>
            <ParamButton onClick={() => preset(80, 64, 8)}>Llama 2 70B</ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="multi-head" value={`${at(heads)} GiB`} />
          <Readout label="grouped-query" value={`${at(kvGroups)} GiB`} />
          <Readout label="multi-query" value={`${at(1)} GiB`} />
          <Readout label="grouped-query per token" value={`${formatNumber(perToken)} KiB`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="tokens cached per sequence"
        yLabel="cache size (GiB)"
        xRange={[0, MAX_LEN]}
        yRange={[0, undefined]}
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
