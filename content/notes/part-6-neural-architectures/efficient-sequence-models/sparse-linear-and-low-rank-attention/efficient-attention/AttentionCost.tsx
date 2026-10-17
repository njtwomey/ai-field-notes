import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Plots, Readout, useAxis, useFigureState } from 'aifn-render'
import { type AttentionMethod, attentionCost, layerLinearCost } from 'aifn-compute/nn/attention'

const METHODS: { method: AttentionMethod; name: string; slot: number }[] = [
  { method: 'full', name: 'full attention', slot: 0 },
  { method: 'flash', name: 'FlashAttention', slot: 1 },
  { method: 'window', name: 'sliding window', slot: 2 },
  { method: 'block', name: 'block selection', slot: 3 },
  { method: 'linear', name: 'linear attention', slot: 4 },
  { method: 'linformer', name: 'Linformer', slot: 5 },
]

const LENGTHS = Array.from({ length: 49 }, (_, i) => Math.round(2 ** (9 + i / 4)))

/** Bytes as KiB, MiB or GiB. */
function bytes(b: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let u = 0
  while (b >= 1024 && u < units.length - 1) {
    b /= 1024
    u++
  }
  return `${formatNumber(Math.round(b * 10) / 10)} ${units[u]}`
}

/** Multiply–adds of one layer, and score memory of one head, against sequence length for each kind of attention. */
export function AttentionCost() {
  const state = useFigureState({
    n: int(8192, { min: 512, max: 1048576, suggestions: [2048, 8192, 32768, 131072], label: 'sequence length n' }),
    dModel: int(2048, { min: 256, max: 16384, suggestions: [1024, 2048, 4096, 8192], label: 'layer width d_model' }),
    d: int(128, { min: 16, max: 256, suggestions: [64, 128], label: 'head width d' }),
    w: int(4096, { min: 64, max: 65536, suggestions: [512, 1024, 4096], label: 'window w' }),
    topK: int(16, { min: 1, max: 64, suggestions: [4, 8, 16], label: 'blocks of 64 kept k' }),
    rank: int(256, { min: 16, max: 4096, suggestions: [128, 256, 512], label: 'Linformer rank k' }),
  })
  const { n, dModel, d, w, topK, rank } = state
  const heads = Math.max(1, Math.round(dModel / d))
  const options = useMemo(
    () => ({ headWidth: d, window: w, blockSize: 64, topK, rank, features: d }),
    [d, w, topK, rank],
  )

  const curves = useMemo(
    () =>
      METHODS.map((m) => ({
        ...m,
        flops: LENGTHS.map((L) => heads * attentionCost(m.method, L, options).multiplyAdds),
        memory: LENGTHS.map((L) => 2 * attentionCost(m.method, L, options).scoreMemory),
      })),
    [heads, options],
  )
  const linear = useMemo(() => LENGTHS.map((L) => layerLinearCost(L, dModel)), [dModel])

  const at = (m: AttentionMethod) => attentionCost(m, n, options)
  const fullFlops = heads * at('full').multiplyAdds
  const share = fullFlops / (fullFlops + layerLinearCost(n, dModel))

  const nAxis = useAxis({ label: 'sequence length n', log: true })
  const flopAxis = useAxis({ label: 'multiply–adds per layer', log: true })
  const memAxis = useAxis({ label: 'score memory per head (bytes, 16-bit)', log: true })
  const drag = (x: number) => state.set('n', Math.round(Math.min(1048576, Math.max(512, x))))

  return (
    <Figure
      title="What attention costs as the sequence grows"
      state={state}
      caption="Left: multiply–adds of one transformer layer's attention (all heads, one sequence, forward pass) against the sequence length n, on log–log axes, with the projections and feed-forward network of the same layer as the dashed line. Full attention's arithmetic, which FlashAttention shares, grows as n² and overtakes the rest of the layer at n = 6 d_model. Windows, block selection, linear attention and Linformer grow as n. Right: the attention scores or running statistics one head holds at once. Full attention stores the n × n matrix; FlashAttention keeps two numbers per query; linear attention keeps a d × d state whatever n is. Drag the vertical line to set n; set the layer and head widths, the window, the number of 64-token blocks kept per query and Linformer's rank."
      readouts={
        <>
          <Readout
            label="attention share of the layer (full)"
            value={`${formatNumber(Math.round(1000 * share) / 10)}%`}
          />
          <Readout label="crossover n = 6 d_model" value={formatNumber(6 * dModel)} />
          <Readout label="full score matrix per head" value={bytes(2 * at('full').scoreMemory)} />
          <Readout
            label="full ÷ window, block, linear"
            value={['window', 'block', 'linear']
              .map(
                (m) => `${formatNumber(Math.round(at('full').multiplyAdds / at(m as AttentionMethod).multiplyAdds))}×`,
              )
              .join(', ')}
          />
          <Readout label="heads" value={String(heads)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={nAxis} y={flopAxis} title="arithmetic">
          <Curve name="projections + feed-forward" x={LENGTHS} y={linear} emphasis dashed />
          {curves
            .filter((c) => c.method !== 'flash')
            .map((c) => (
              <Curve key={c.method} name={c.name} x={LENGTHS} y={c.flops} slot={c.slot} />
            ))}
          <Handle kind="x" at={n} label="n" onDrag={drag} />
        </Plot>
        <Plot x={nAxis} y={memAxis} title="score memory">
          {curves.map((c) => (
            <Curve key={c.method} name={c.name} x={LENGTHS} y={c.memory} slot={c.slot} />
          ))}
          <Handle kind="x" at={n} label="n" onDrag={drag} />
        </Plot>
      </Plots>
    </Figure>
  )
}
