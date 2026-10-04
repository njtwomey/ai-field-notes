import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Plots,
  Plot,
  Bars,
  Points,
  Annotation,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { decodeThroughput, servingMemory } from 'aifn-applied/neural/quantisation'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')
const SERVE_BITS = [16, 8, 6, 4, 3, 2]

export function ServingCostExplorer() {
  const [billions, setBillions] = useState(7)
  const [layers, setLayers] = useState(32)
  const [kvHeads, setKvHeads] = useState(8)
  const [headDim, setHeadDim] = useState(128)

  const [context, setContext] = useState(4096)
  const [batch, setBatch] = useState(1)
  const [kvBits, setKvBits] = useState(16)

  const [memoryGb, setMemoryGb] = useState(24)
  const [bandwidthGbs, setBandwidthGbs] = useState(1000)
  const [tflops, setTflops] = useState(150)

  const shape = useMemo(
    () => ({
      parameters: billions * 1e9,
      layers,
      kvHeads,
      headDim,
    }),
    [billions, layers, kvHeads, headDim],
  )

  const device = useMemo(
    () => ({
      memoryGb,
      bandwidthGbs,
      tflops,
    }),
    [memoryGb, bandwidthGbs, tflops],
  )

  const rows = useMemo(
    () =>
      SERVE_BITS.map((b) => {
        const setup = {
          weightBits: b,
          kvBits,
          batch,
          context,
        }
        return {
          bits: b,
          mem: servingMemory(shape, setup, device),
          speed: decodeThroughput(shape, setup, device),
        }
      }),
    [shape, kvBits, batch, context, device],
  )

  const fourBit = rows[3] // index 3 corresponds to 4 bits

  const xb = useAxis({ label: 'weight precision', categories: SERVE_BITS.map((b) => `${b}-bit`) })
  const ym = useAxis({ label: 'memory consumption (GB)', range: [0, undefined], hold: 'union' })
  const yt = useAxis({ label: 'decoding speed (tokens/sec)', range: [0, undefined], hold: 'union' })

  return (
    <Figure
      title="Serving economics: memory footprint and decoding speed vs bit width"
      purpose="During autoregressive generation, memory bandwidth bounds throughput at small batches: halving bits halves bytes loaded per token, roughly doubling decoding speed until arithmetic throughput saturates."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="1 · Model architecture">
            <NumberSelector
              label="parameters (billions)"
              value={billions}
              onChange={setBillions}
              min={1}
              max={70}
              step={1}
              suggestions={[1, 7, 14, 70]}
            />
            <NumberSelector
              label="layers"
              value={layers}
              onChange={setLayers}
              min={12}
              max={80}
              step={4}
              suggestions={[24, 32, 48, 80]}
            />
            <NumberSelector
              label="KV heads"
              value={kvHeads}
              onChange={setKvHeads}
              min={1}
              max={32}
              step={1}
              suggestions={[4, 8, 16, 32]}
            />
            <NumberSelector
              label="head dimension"
              value={headDim}
              onChange={setHeadDim}
              min={64}
              max={256}
              step={32}
              suggestions={[64, 128]}
            />
          </ControlGroup>
          <ControlGroup title="2 · Serving configuration">
            <NumberSelector
              label="batch size"
              value={batch}
              onChange={setBatch}
              min={1}
              max={128}
              step={1}
              suggestions={[1, 4, 16, 64]}
            />
            <NumberSelector
              label="context length (tokens)"
              value={context}
              onChange={setContext}
              min={512}
              max={32768}
              step={1024}
              suggestions={[2048, 4096, 8192, 16384]}
            />
            <Select
              label="KV cache precision"
              value={String(kvBits)}
              onChange={(v) => setKvBits(Number(v))}
              options={[
                { value: '16', label: '16-bit (FP16 / BF16)' },
                { value: '8', label: '8-bit (FP8 / INT8)' },
                { value: '4', label: '4-bit (INT4)' },
              ]}
            />
          </ControlGroup>
          <ControlGroup title="3 · Hardware device specs">
            <NumberSelector
              label="device VRAM (GB)"
              value={memoryGb}
              onChange={setMemoryGb}
              min={8}
              max={192}
              step={4}
              suggestions={[16, 24, 48, 80, 141]}
            />
            <NumberSelector
              label="memory bandwidth (GB/s)"
              value={bandwidthGbs}
              onChange={setBandwidthGbs}
              min={200}
              max={4000}
              step={100}
              suggestions={[400, 1000, 2000, 3350]}
            />
            <NumberSelector
              label="dense TFLOP/s"
              value={tflops}
              onChange={setTflops}
              min={50}
              max={2000}
              step={50}
              suggestions={[150, 300, 990]}
            />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="4-bit weights memory" value={`${fmt(fourBit.mem.weights / 1e9)} GB`} />
          <Readout label="4-bit KV cache memory" value={`${fmt(fourBit.mem.kvCache / 1e9)} GB`} />
          <Readout label="fits on single device" value={fourBit.mem.fits ? 'yes' : 'no'} />
          <Readout label="bottleneck regime" value={fourBit.speed.bound} />
          <Readout label="ridge batch (compute limit)" value={fmt(fourBit.speed.ridgeBatch)} />
        </>
      }
      caption="Left: Memory breakdown for weights (blue) and KV cache (orange) across weight bit widths (16-bit down to 2-bit); the dashed horizontal rule marks device VRAM. Right: Roofline decoding speed (tokens per second) bounded by memory bandwidth at low batch and arithmetic compute at high batch. Notice how 4-bit quantisation allows models that exceed device VRAM in FP16 to fit cleanly on a single GPU."
    >
      <Plots cols={2}>
        <Plot x={xb} y={ym} title="VRAM allocation (weights vs KV cache)">
          <Bars
            name="model weights"
            x={SERVE_BITS.map((_, idx) => idx - 0.2)}
            y={rows.map((r) => r.mem.weights / 1e9)}
            width={0.38}
            slot={0}
          />
          <Bars
            name="KV cache"
            x={SERVE_BITS.map((_, idx) => idx + 0.2)}
            y={rows.map((r) => r.mem.kvCache / 1e9)}
            width={0.38}
            slot={1}
          />
          <Annotation y={device.memoryGb} dashed text={`device memory: ${device.memoryGb} GB`} />
        </Plot>
        <Plot x={xb} y={yt} title="Decoding throughput (tokens/sec roofline)">
          <Bars
            name="tokens / sec"
            x={SERVE_BITS.map((_, idx) => idx)}
            y={rows.map((r) => r.speed.tokensPerSecond)}
            slot={2}
          />
          <Points
            name="compute-bound regime"
            x={rows.flatMap((r, idx) => (r.speed.bound === 'compute' ? [idx] : []))}
            y={rows.flatMap((r) => (r.speed.bound === 'compute' ? [r.speed.tokensPerSecond] : []))}
            emphasis
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
