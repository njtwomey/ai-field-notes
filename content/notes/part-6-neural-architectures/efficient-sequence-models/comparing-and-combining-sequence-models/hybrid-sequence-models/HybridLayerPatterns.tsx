import { useMemo } from 'react'
import { choice, Diagram, Figure, formatNumber, int, Readout, useFigureState } from 'aifn-render'
import type { DiagramEdge, DiagramGroup, DiagramNode, DiagramSpec, Tone } from 'aifn-render'
import { kvCacheMemory } from 'aifn-compute/nn/attention'

/** What a token-mixing layer is: a recurrence, global (full) attention, or sliding-window attention. */
type Kind = 'recurrent' | 'global' | 'window'

const TONE: Record<Kind, Tone> = { recurrent: 0, global: 1, window: 2 }

/** One box of the zoomed view: a token mixer (with its kind) or a feed-forward sublayer (`ffn`). */
type Part = { label: string; kind: Kind | 'ffn'; dashed?: boolean }

type Model = {
  name: string
  /** The token mixers of the whole stack, in order, or of one repeating unit when the depth is not published. */
  stack: Kind[]
  /** The stack above is one unit repeated an unknown number of times. */
  unitOnly?: boolean
  /** Window of the sliding-window layers, in tokens. */
  window?: number
  /** The zoomed stretch, left to right. */
  zoom: Part[]
  zoomLabel: string
  /** A custom zoom (Hymba's parallel heads). */
  parallel?: boolean
  facts: string
}

const repeat = <T,>(n: number, f: (i: number) => T) => Array.from({ length: n }, (_, i) => f(i))
const NEMOTRON = 'M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-'
const ZAMBA_HYBRID = new Set([3, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74])

const MODELS = {
  jamba: {
    name: 'Jamba',
    stack: repeat<Kind>(32, (i) => (i % 8 === 4 ? 'global' : 'recurrent')),
    zoom: repeat<Part>(8, (i) => ({
      label: `${i === 4 ? 'attention' : 'Mamba'}\n+ ${i % 2 === 1 ? 'MoE' : 'MLP'}`,
      kind: i === 4 ? 'global' : 'recurrent',
    })),
    zoomLabel: 'one Jamba block: 8 layers, attention fifth, MoE on every second layer',
    facts: '32 layers in 4 blocks of 8; 1 attention : 7 Mamba; 16 experts, top 2',
  },
  griffin: {
    name: 'Griffin',
    stack: ['recurrent', 'recurrent', 'window'],
    unitOnly: true,
    window: 1024,
    zoom: [
      { label: 'RG-LRU\n+ MLP', kind: 'recurrent' },
      { label: 'RG-LRU\n+ MLP', kind: 'recurrent' },
      { label: 'local MQA\n+ MLP', kind: 'window' },
    ],
    zoomLabel: 'repeating unit: two recurrent blocks, one local-attention block',
    facts: '2 RG-LRU : 1 local attention (window 1024); Hawk is RG-LRU only',
  },
  zamba: {
    name: 'Zamba',
    stack: repeat<Kind>(76, (i) => (ZAMBA_HYBRID.has(i + 1) ? 'global' : 'recurrent')),
    zoom: [
      ...repeat<Part>(6, () => ({ label: 'Mamba', kind: 'recurrent' as const })),
      { label: 'shared attention\n+ MLP', kind: 'global', dashed: true },
    ],
    zoomLabel: 'every 6 Mamba blocks, one call of the shared attention block (dashed: same weights each time)',
    facts: '76 Mamba layers; one attention + MLP block, weights shared, called at 13 of them',
  },
  samba: {
    name: 'Samba',
    stack: repeat<Kind>(32, (i) => (i % 2 === 1 ? 'window' : 'recurrent')),
    window: 2048,
    zoom: [
      { label: 'Mamba', kind: 'recurrent' },
      { label: 'MLP', kind: 'ffn' },
      { label: 'sliding-window\nattention', kind: 'window' },
      { label: 'MLP', kind: 'ffn' },
    ],
    zoomLabel: 'repeating unit: Mamba, MLP, sliding-window attention, MLP',
    facts: '3.8B model: 64 layers, 16 Mamba, 16 SWA (window 2048), 32 MLP',
  },
  hymba: {
    name: 'Hymba',
    stack: repeat<Kind>(32, (i) => ([0, 15, 31].includes(i) ? 'global' : 'window')),
    window: 1024,
    zoom: [],
    parallel: true,
    zoomLabel: 'one Hymba layer: attention heads and SSM heads read the same input in parallel',
    facts:
      '32 layers, each attention ∥ SSM; global attention in layers 1, 16, 32, the rest window 1024; 128 meta tokens',
  },
  nemotron: {
    name: 'Nemotron-H-8B',
    stack: [...NEMOTRON].filter((c) => c !== '-').map((c): Kind => (c === '*' ? 'global' : 'recurrent')),
    zoom: [...NEMOTRON.slice(0, 9)].map((c): Part =>
      c === 'M'
        ? { label: 'Mamba-2', kind: 'recurrent' as const }
        : c === '*'
          ? { label: 'attention', kind: 'global' as const }
          : { label: 'FFN', kind: 'ffn' as const },
    ),
    zoomLabel: 'first 9 of 52 layers: Mamba-2 and FFN layers, attention before an FFN',
    facts: '52 layers: 24 Mamba-2, 4 attention, 24 FFN (about 8% attention)',
  },
  minimax: {
    name: 'MiniMax-01',
    stack: repeat<Kind>(80, (i) => (i % 8 === 7 ? 'global' : 'recurrent')),
    zoom: repeat<Part>(8, (i) => ({
      label: `${i === 7 ? 'softmax\nattention' : 'lightning\nattention'}\n+ MoE`,
      kind: i === 7 ? 'global' : 'recurrent',
    })),
    zoomLabel: 'repeating unit: 7 linear (lightning) attention layers, then 1 softmax attention layer',
    facts: '80 layers, 7 linear : 1 softmax; 456B parameters, 45.9B active, 32 experts',
  },
  qwen: {
    name: 'Qwen3-Next',
    stack: repeat<Kind>(48, (i) => (i % 4 === 3 ? 'global' : 'recurrent')),
    zoom: [
      ...repeat<Part>(3, () => ({ label: 'Gated DeltaNet\n+ MoE', kind: 'recurrent' as const })),
      { label: 'gated attention\n+ MoE', kind: 'global' },
    ],
    zoomLabel: 'repeating unit, 12 times: 3 Gated DeltaNet layers, 1 gated attention layer',
    facts: '48 layers, 3 Gated DeltaNet : 1 attention; 80B parameters, 3B active',
  },
  kimi: {
    name: 'Kimi Linear',
    stack: ['recurrent', 'recurrent', 'recurrent', 'global'],
    unitOnly: true,
    zoom: [
      ...repeat<Part>(3, () => ({ label: 'KDA\n+ MoE', kind: 'recurrent' as const })),
      { label: 'MLA (no positions)\n+ MoE', kind: 'global' },
    ],
    zoomLabel: 'repeating unit: 3 Kimi Delta Attention layers, 1 multi-head latent attention layer',
    facts: '3 KDA : 1 MLA; 48B parameters, 3B active',
  },
} satisfies Record<string, Model>

type ModelId = keyof typeof MODELS

const OPTIONS = (Object.keys(MODELS) as ModelId[]).map((id) => ({ value: id, label: MODELS[id].name }))

const PER_ROW = 16
const GAP = 0.95
const BOX = 0.8

function buildSpec(model: Model): DiagramSpec {
  const nodes: DiagramNode[] = []
  const edges: DiagramEdge[] = []
  // The whole stack (or 4 repeats of the unit), one small box per token-mixing layer, 16 to a row.
  const stack = model.unitOnly ? repeat(4, () => model.stack).flat() : model.stack
  stack.forEach((kind, i) => {
    nodes.push({
      id: `s${i}`,
      x: (i % PER_ROW) * GAP,
      y: Math.floor(i / PER_ROW) * GAP,
      w: BOX,
      h: BOX,
      tone: TONE[kind],
      ariaLabel: `layer ${i + 1}: ${kind}`,
    })
  })
  const rows = Math.ceil(stack.length / PER_ROW)
  if (model.unitOnly)
    nodes.push({
      id: 'more',
      x: (stack.length % PER_ROW) * GAP + 0.6,
      y: (rows - 1) * GAP,
      shape: 'text',
      w: 2.4,
      label: '… repeated',
      small: true,
    })
  const groups: DiagramGroup[] = [
    {
      id: 'stack',
      around: stack.map((_, i) => `s${i}`),
      pad: 0.35,
      label: model.unitOnly
        ? 'the unit repeated (depth not published)'
        : `whole stack: ${stack.length} token-mixing layers`,
      labelAt: 'top-left',
      tone: 'neutral',
    },
  ]

  // The zoomed stretch below.
  const top = rows * GAP + 1.6
  if (model.parallel) {
    const cx = 6.6
    nodes.push(
      { id: 'zin', x: 0.6, y: top + 1.1, shape: 'text', w: 1.6, label: 'input\n+ meta tokens', small: true },
      {
        id: 'za',
        x: cx - 1.6,
        y: top + 0.4,
        w: 3,
        h: 0.8,
        label: 'attention heads\n(global or window)',
        tone: 1,
        small: true,
      },
      { id: 'zs', x: cx - 1.6, y: top + 1.8, w: 3, h: 0.8, label: 'SSM heads (Mamba)', tone: 0, small: true },
      {
        id: 'zm',
        x: cx + 1.9,
        y: top + 1.1,
        w: 2.4,
        h: 0.9,
        label: 'normalise, scale,\naverage',
        tone: 'neutral',
        small: true,
      },
      { id: 'zf', x: cx + 4.6, y: top + 1.1, w: 1.4, h: 0.8, label: 'FFN', tone: 'neutral', small: true },
    )
    edges.push(
      { from: 'zin', to: 'za:w' },
      { from: 'zin', to: 'zs:w' },
      { from: 'za:e', to: 'zm' },
      { from: 'zs:e', to: 'zm' },
      { from: 'zm', to: 'zf' },
    )
  } else {
    const w = 1.55
    model.zoom.forEach((part, i) => {
      nodes.push({
        id: `z${i}`,
        x: i * (w + 0.25),
        y: top + 1.1,
        w,
        h: 1.1,
        label: part.label,
        tone: part.kind === 'ffn' ? 'neutral' : TONE[part.kind],
        dashed: part.dashed,
        small: true,
      })
      if (i > 0) edges.push({ from: `z${i - 1}`, to: `z${i}` })
    })
    if (model === MODELS.zamba) {
      const last = model.zoom.length - 1
      nodes.push({
        id: 'zx0',
        x: last * (w + 0.25),
        y: top + 2.6,
        shape: 'text',
        w: 2.6,
        label: 'input embeddings $x_0$',
        small: true,
      })
      edges.push({ from: 'zx0', to: `z${last}:s`, label: 'concatenated', labelRotate: false })
    }
  }
  groups.push({
    id: 'zoom',
    around: nodes.filter((n) => n.id.startsWith('z') && n.shape !== 'text').map((n) => n.id),
    pad: 0.35,
    label: model.zoomLabel,
    labelAt: 'top-left',
    tone: 'neutral',
  })
  return { nodes, edges, groups }
}

/** The fraction of a full-attention model's KV cache that a hybrid keeps, at a given context length. */
function cacheShare(model: Model, context: number) {
  const global = model.stack.filter((k) => k === 'global').length
  const window = model.stack.filter((k) => k === 'window').length
  const layout = { heads: 1, headDim: 1 }
  const full = kvCacheMemory({ ...layout, layers: model.stack.length }, context)
  const kept =
    (global > 0 ? kvCacheMemory({ ...layout, layers: global }, context).bytes : 0) +
    (window > 0 ? kvCacheMemory({ ...layout, layers: window, window: model.window }, context).bytes : 0)
  return { global, window, mixers: model.stack.length, share: kept / full.bytes }
}

/** Layer patterns of published hybrid sequence models: the whole stack and a zoomed unit. */
export function HybridLayerPatterns() {
  const state = useFigureState({
    model: choice<ModelId>(OPTIONS, 'jamba', { label: 'model' }),
    context: int(131072, {
      min: 1024,
      max: 4194304,
      label: 'context length (tokens)',
      suggestions: [4096, 32768, 131072, 1048576],
    }),
  })
  const model: Model = MODELS[state.model]
  const spec = useMemo(() => buildSpec(model), [model])
  const cache = cacheShare(model, state.context)
  const unit = model.unitOnly ? ' per unit' : ''

  return (
    <Figure
      title="Layer patterns of hybrid sequence models"
      state={state}
      caption="Choose a model. Top: every token-mixing layer of the stack in order, 16 to a row (for Griffin and Kimi Linear, whose depth is not given in their papers, the repeating unit four times). Blue boxes are recurrent or linear-attention layers, which keep a fixed-size state; orange boxes are global attention and green boxes sliding-window attention, which keep a key-value cache. Hymba's layers each run attention and SSM heads side by side, so its stack is coloured by the attention part. Bottom: one stretch of the stack with its feed-forward sublayers. The readouts compare the key-value cache with a model of the same depth in which every token mixer is global attention with the same heads, at the context length typed in: global layers keep every token, window layers at most their window, recurrent layers none. Raise the context length to see the window hybrids' share fall towards the share of global layers."
      readouts={
        <>
          <Readout label="model" value={model.facts} />
          <Readout label={`global attention layers${unit}`} value={`${cache.global} of ${cache.mixers}`} />
          <Readout label={`sliding-window layers${unit}`} value={`${cache.window} of ${cache.mixers}`} />
          <Readout label="KV cache vs all-attention stack" value={`${formatNumber(100 * cache.share)}%`} />
        </>
      }
    >
      <Diagram spec={spec} ariaLabel={`Layer pattern of ${model.name}: ${model.zoomLabel}`} />
    </Figure>
  )
}
