import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

// Llama 2 7B dimensions: width, heads, head size, layers.
const D = 4096
const HEADS = 32
const DH = 128
const LAYERS = 32
const BYTES = 2 // fp16 or bf16 cache entries

type Config = {
  placement: 'post' | 'pre'
  norm: 'ln' | 'rms'
  ffn: 'relu' | 'swiglu'
  attn: 'mha' | 'gqa' | 'mqa'
  position: 'sinusoidal' | 'rope'
  bias: 'yes' | 'no'
}

const ORIGINAL: Config = {
  placement: 'post',
  norm: 'ln',
  ffn: 'relu',
  attn: 'mha',
  position: 'sinusoidal',
  bias: 'yes',
}
const MODERN: Config = { placement: 'pre', norm: 'rms', ffn: 'swiglu', attn: 'gqa', position: 'rope', bias: 'no' }

const kvHeads = (c: Config) => (c.attn === 'mha' ? HEADS : c.attn === 'gqa' ? 8 : 1)
const hidden = (c: Config) => (c.ffn === 'relu' ? 4 * D : 11008)

/** Weight and bias counts of one block, by part. */
function parts(c: Config) {
  const kv = kvHeads(c) * DH
  const f = hidden(c)
  const attention = 2 * D * D + 2 * D * kv
  const feedForward = (c.ffn === 'relu' ? 2 : 3) * D * f
  const norms = 2 * (c.norm === 'ln' ? 2 * D : D)
  const biases = c.bias === 'yes' ? D + 2 * kv + D + (c.ffn === 'relu' ? f : 2 * f) + D : 0
  return { attention, feedForward, norms, biases, total: attention + feedForward + norms + biases }
}

/** Matrix-product FLOPs for one new token at context n, in one block: 2 per weight plus 4nd for scores and values. */
const flops = (c: Config, n: number) => 2 * (parts(c).attention + parts(c).feedForward) + 4 * n * D
/** Key and value bytes cached per token across all layers. */
const kvBytes = (c: Config) => 2 * kvHeads(c) * DH * BYTES * LAYERS

const millions = (v: number) => `${(v / 1e6).toFixed(v < 1e6 ? 3 : 1)} M`
const CONTEXTS = Array.from({ length: 64 }, (_, i) => 512 * (i + 1))

type Key = keyof Config
const CONTROLS: { key: Key; label: string; options: { value: string; label: string }[] }[] = [
  {
    key: 'placement',
    label: 'normalisation placement',
    options: [
      { value: 'post', label: 'post (2017)' },
      { value: 'pre', label: 'pre' },
    ],
  },
  {
    key: 'norm',
    label: 'normalisation',
    options: [
      { value: 'ln', label: 'LayerNorm (2017)' },
      { value: 'rms', label: 'RMSNorm' },
    ],
  },
  {
    key: 'ffn',
    label: 'feed-forward',
    options: [
      { value: 'relu', label: 'ReLU, 4d (2017)' },
      { value: 'swiglu', label: 'SwiGLU, 11008' },
    ],
  },
  {
    key: 'attn',
    label: 'key-value heads',
    options: [
      { value: 'mha', label: '32, MHA (2017)' },
      { value: 'gqa', label: '8, GQA' },
      { value: 'mqa', label: '1, MQA' },
    ],
  },
  {
    key: 'position',
    label: 'position',
    options: [
      { value: 'sinusoidal', label: 'sinusoidal (2017)' },
      { value: 'rope', label: 'RoPE' },
    ],
  },
  {
    key: 'bias',
    label: 'biases',
    options: [
      { value: 'yes', label: 'yes (2017)' },
      { value: 'no', label: 'no' },
    ],
  },
]

/** The choice field for one component, starting from the modern choice. */
function component<K extends Key>(key: K) {
  const c = CONTROLS.find((x) => x.key === key)!
  return choice<Config[K]>(c.options as { value: Config[K]; label: string }[], MODERN[key], { label: c.label })
}

/** Toggle each component of a d = 4096 decoder block between the 2017 choice and the modern one. */
export function BlockBuilder() {
  const state = useFigureState({
    placement: component('placement'),
    norm: component('norm'),
    ffn: component('ffn'),
    attn: component('attn'),
    position: component('position'),
    bias: component('bias'),
    n: int(4096, { min: 512, max: 32768, step: 512, label: 'context n (tokens)', format: (v) => v.toFixed(0) }),
    metric: choice<'flops' | 'kv'>(
      [
        { value: 'kv', label: 'KV cache, GiB' },
        { value: 'flops', label: 'GFLOPs per token' },
      ],
      'kv',
      { label: 'plot' },
    ),
  })
  const { placement, norm, ffn, attn, position, bias } = state
  const config: Config = useMemo(
    () => ({ placement, norm, ffn, attn, position, bias }),
    [placement, norm, ffn, attn, position, bias],
  )
  const p = useMemo(() => parts(config), [config])
  const o = useMemo(() => parts(ORIGINAL), [])

  const series = useMemo(() => {
    const value = (c: Config, ctx: number) =>
      state.metric === 'kv' ? (kvBytes(c) * ctx) / 2 ** 30 : (LAYERS * flops(c, ctx)) / 1e9
    return [
      {
        name: '2017 choices',
        x: CONTEXTS,
        y: CONTEXTS.map((ctx) => value(ORIGINAL, ctx)),
        dashed: true,
        slot: 1,
      },
      { name: 'current selection', x: CONTEXTS, y: CONTEXTS.map((ctx) => value(config, ctx)), slot: 0 },
    ] as const
  }, [config, state.metric])

  const rows: [string, number, number, string][] = [
    ['attention projections', o.attention, p.attention, `${kvHeads(config)} key-value heads of size ${DH}`],
    [
      'feed-forward',
      o.feedForward,
      p.feedForward,
      config.ffn === 'relu' ? 'two matrices, 16384 wide' : 'three matrices, 11008 wide',
    ],
    [
      'normalisation gains and shifts',
      o.norms,
      p.norms,
      config.norm === 'ln' ? '2 layer norms × 2d' : '2 RMSNorms × d',
    ],
    ['biases', o.biases, p.biases, config.bias === 'yes' ? 'every linear map' : 'none'],
    ['position', 0, 0, config.position === 'rope' ? 'rotates q and k in every layer' : 'added once to the input'],
  ]

  const xAxis = useAxis({ label: 'context length n', hold: 'union' })
  const yAxis = useAxis({
    label: state.metric === 'kv' ? 'KV cache per sequence (GiB)' : 'GFLOPs per new token',
    hold: 'union',
  })
  return (
    <Figure
      title="Decoder block builder"
      state={state}
      caption="A decoder block with width d = 4096, 32 query heads of size 128 and 32 layers, the dimensions of Llama 2 7B (which keeps 32 key-value heads; Llama 3 8B uses 8). Each control switches one component between the 2017 choice and the modern one. Parameters are per block. FLOPs count two per weight in the matrix products plus 4nd for the scores and the weighted sum of values when the new token attends to n earlier tokens; normalisations, activations and rotations are elementwise and not counted. The KV cache stores one key and one value vector per key-value head per layer at 2 bytes per number. Placement and position change no count: their effects are on training stability and on how position enters."
      readouts={
        <>
          <Readout label="parameters per block" value={millions(p.total)} />
          <Readout label="GFLOPs per token (32 blocks)" value={((LAYERS * flops(config, state.n)) / 1e9).toFixed(2)} />
          <Readout label="KV cache per token" value={`${(kvBytes(config) / 1024).toFixed(0)} KiB`} />
          <Readout label="KV cache at n" value={`${((kvBytes(config) * state.n) / 2 ** 30).toFixed(2)} GiB`} />
          <Readout label="warmup" value={config.placement === 'post' ? 'required' : 'optional'} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
        </Plot>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-muted-foreground">
              <tr className="border-b">
                <th className="py-1 text-left font-normal">part of one block</th>
                <th className="py-1 text-right font-normal">2017</th>
                <th className="py-1 text-right font-normal">selected</th>
                <th className="py-1 pl-4 text-left font-normal">selected form</th>
              </tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {rows.map(([name, a, b, note]) => (
                <tr key={name} className="border-b border-border/50">
                  <td className="py-1 font-sans">{name}</td>
                  <td className="py-1 text-right">{millions(a)}</td>
                  <td className="py-1 text-right">{millions(b)}</td>
                  <td className="py-1 pl-4 font-sans text-muted-foreground">{note}</td>
                </tr>
              ))}
              <tr>
                <td className="py-1 font-sans">total</td>
                <td className="py-1 text-right">{millions(o.total)}</td>
                <td className="py-1 text-right">{millions(p.total)}</td>
                <td />
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </Figure>
  )
}
