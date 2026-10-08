/**
 * Diagram specs for the parameter-efficient fine-tuning note. Every diagram flows left to right and uses one
 * convention: frozen pretrained parts are neutral, trainable parts are drawn in the accent colour with a heavier
 * outline (`highlight`), and operations without parameters (sums, products, the nonlinearity) are plain.
 */
import { op, projector } from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec } from 'aifn-render'

type Extra = Partial<DiagramNode>

/** A frozen pretrained component. */
const frozen = (id: string, x: number, y: number, label: string, extra: Extra = {}): DiagramNode => ({
  id,
  x,
  y,
  label,
  tone: 'neutral',
  w: 1,
  h: 0.6,
  ...extra,
})

/** A trainable component. */
const trainable = (id: string, x: number, y: number, label: string, extra: Extra = {}): DiagramNode => ({
  id,
  x,
  y,
  label,
  highlight: true,
  w: 1,
  h: 0.6,
  ...extra,
})

const text = (id: string, x: number, y: number, label: string, extra: Extra = {}): DiagramNode => ({
  id,
  x,
  y,
  shape: 'text',
  label,
  ...extra,
})

const dot = (id: string, x: number, y: number, extra: Extra = {}): DiagramNode => ({ id, x, y, shape: 'dot', ...extra })

/** An invisible point on a node's outline, so several edges can arrive at one box at different heights. */
const anchor = (id: string, x: number, y: number): DiagramNode => ({ id, x, y, shape: 'text', w: 0.02, h: 0.02 })

/** A method's name beside the place it attaches, joined to it by a dashed accent line. */
const marker = (id: string, x: number, y: number, label: string): DiagramNode => ({
  id,
  x,
  y,
  shape: 'pill',
  label,
  highlight: true,
  small: true,
  h: 0.5,
  w: 1.2,
})
const pointer = (from: string, to: string, via?: [number, number][]): DiagramEdge => ({
  from,
  to,
  via,
  dashed: true,
  tone: 0,
})

/** Rounded to a thousandth, so that points meant to share a line compare equal and edges between them stay straight. */
const snap = (v: number) => Math.round(v * 1000) / 1000

/** Vertical distance between the bars of a sequence. */
const PITCH = 0.36

/**
 * A column of `n` bars, one per sequence position (each a vector of width d), the first `lead` of them trainable. The
 * column runs down from `y0`.
 */
function cells(p: string, x: number, y0: number, n: number, lead: number): DiagramNode[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `${p}${i}`,
    x,
    y: snap(y0 + PITCH * i),
    w: 0.9,
    h: 0.26,
    ...(i < lead ? { highlight: true } : { tone: 'neutral' as const }),
  }))
}

/**
 * A 4 × 4 grid standing for the entries of a weight matrix, with top-left entry centred at (x0, y0). Without `lit`
 * every entry is a frozen square; with `lit` the listed entries are trainable squares and the rest are dots (zeros).
 */
function grid(p: string, x0: number, y0: number, lit?: [number, number][]): DiagramNode[] {
  const out: DiagramNode[] = []
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      const at = { id: `${p}${i}${j}`, x: snap(x0 + 0.48 * j), y: snap(y0 + 0.48 * i) }
      if (!lit) out.push({ ...at, w: 0.4, h: 0.4, tone: 'neutral' })
      else if (lit.some(([a, b]) => a === i && b === j)) out.push({ ...at, w: 0.4, h: 0.4, highlight: true })
      else out.push({ ...at, shape: 'dot', w: 0.1, h: 0.1, tone: 'neutral' })
    }
  return out
}

/** Row of the feed-forward sublayer in the overview, below the attention sublayer. */
const Y = 5.75

/**
 * One pre-norm transformer block with a gated feed-forward network, and where each method attaches. The attention
 * sublayer is the top row; the residual stream wraps round to the feed-forward sublayer in the bottom row.
 */
export const overview: DiagramSpec = {
  unit: 40,
  nodes: [
    text('in', 0, 0, '$\\xvec$', { w: 0.6 }),
    dot('d1', 0.7, 0),
    frozen('ln1', 1.8, 0, 'norm', { w: 1.1 }),
    frozen('wq', 3.4, -1.1, '$\\Wmat_q$', { w: 0.9 }),
    frozen('wk', 3.4, 0, '$\\Wmat_k$', { w: 0.9 }),
    frozen('wv', 3.4, 1.1, '$\\Wmat_v$', { w: 0.9 }),
    frozen('att', 5.3, 0, 'attention', { w: 1.6, h: 2.8 }),
    anchor('aq', 4.5, -1.1),
    anchor('ak', 4.5, 0),
    anchor('av', 4.5, 1.1),
    anchor('mv', 4.17, 1.1),
    frozen('wo', 7.0, 0, '$\\Wmat_o$', { w: 0.9 }),
    anchor('ma1', 7.75, 0),
    op('add1', 8.3, 0, '$+$'),
    dot('d2', 0.7, Y),
    frozen('ln2', 1.8, Y, 'norm', { w: 1.1 }),
    frozen('gate', 3.3, Y - 0.75, 'gate'),
    frozen('up', 3.3, Y + 0.75, 'up'),
    frozen('act', 4.7, Y - 0.75, 'SiLU'),
    op('mul', 5.8, Y, '$\\odot$'),
    anchor('mff', 6.3, Y),
    frozen('down', 7.1, Y, 'down'),
    anchor('ma2', 7.85, Y),
    op('add2', 8.4, Y, '$+$'),
    text('out', 9.2, Y, '$\\hvec$', { w: 0.6 }),
    marker('pPrompt', 0, 1.85, 'prompt tuning'),
    marker('pLora', 3.4, 2.55, 'LoRA'),
    marker('pIa3', 4.17, 1.85, 'IA3'),
    marker('pPrefix', 5.3, 2.55, 'prefix tuning'),
    marker('pBitfit', 7.0, 1.85, 'BitFit'),
    marker('pAd1', 7.75, 2.55, 'adapter'),
    marker('pIa3f', 6.3, 4.05, 'IA3'),
    marker('pAd2', 7.85, 4.05, 'adapter'),
    marker('pReft', 9.2, 4.05, 'ReFT'),
    frozen('keyF', 10.4, -2.0, 'frozen', { small: true, w: 1.3, h: 0.5 }),
    trainable('keyT', 10.4, -1.35, 'trainable', { small: true, w: 1.3, h: 0.5 }),
  ],
  edges: [
    { from: 'in', to: 'd1', arrow: 'none' },
    { from: 'd1', to: 'ln1' },
    { from: 'ln1:e', to: 'wq:w' },
    { from: 'ln1:e', to: 'wk:w' },
    { from: 'ln1:e', to: 'wv:w' },
    { from: 'wq', to: 'aq:w' },
    { from: 'wk', to: 'ak:w' },
    { from: 'wv', to: 'av:w' },
    { from: 'att', to: 'wo' },
    { from: 'wo', to: 'add1' },
    {
      from: 'add1:e',
      to: 'd2:w',
      via: [
        [9.2, 0],
        [9.2, 3.25],
        [-0.4, 3.25],
        [-0.4, Y],
      ],
    },
    { from: 'd2', to: 'ln2' },
    { from: 'ln2:e', to: 'gate:w' },
    { from: 'ln2:e', to: 'up:w' },
    { from: 'gate', to: 'act' },
    { from: 'act:e', to: 'mul:n', via: [[5.8, Y - 0.75]] },
    { from: 'up:e', to: 'mul:s', via: [[5.8, Y + 0.75]] },
    { from: 'mul', to: 'down' },
    { from: 'down', to: 'add2' },
    { from: 'add2', to: 'out' },
    {
      from: 'd1:n',
      to: 'add1:n',
      via: [
        [0.7, -2.0],
        [8.3, -2.0],
      ],
    },
    {
      from: 'd2:s',
      to: 'add2:s',
      via: [
        [0.7, Y + 1.8],
        [8.4, Y + 1.8],
      ],
    },
    pointer('pPrompt:n', 'in:s'),
    pointer('pLora:n', 'wv:s'),
    pointer('pIa3:n', 'mv:s'),
    pointer('pPrefix:n', 'att:s'),
    pointer('pBitfit:n', 'wo:s'),
    pointer('pAd1:n', 'ma1:s'),
    pointer('pIa3f:s', 'mff:n'),
    pointer('pAd2:s', 'ma2:n'),
    pointer('pReft:s', 'out:n'),
  ],
}

/** A Houlsby adapter after one sublayer of a pre-norm block. */
export const adapter: DiagramSpec = {
  unit: 40,
  nodes: [
    text('in', 0, 0, '$\\xvec$', { w: 0.6 }),
    dot('d1', 0.7, 0),
    frozen('ln', 1.8, 0, 'norm', { w: 1.1 }),
    frozen('sub', 3.9, 0, 'attention or\nfeed-forward', { w: 2.2, h: 1.0 }),
    dot('dh', 5.3, 0, { label: '$\\hvec$', labelSide: 's' }),
    {
      ...projector('encoder', 'down', 6.9, 0, '$\\Wmat_{\\text{down}}$', { w: 1.4, h: 1.6 }),
      highlight: true,
      notes: { s: '$d \\to m$' },
    },
    frozen('f', 8.5, 0, '$f$', { w: 0.6 }),
    {
      ...projector('decoder', 'up', 10.1, 0, '$\\Wmat_{\\text{up}}$', { w: 1.4, h: 1.6 }),
      highlight: true,
      notes: { s: '$m \\to d$' },
    },
    op('adda', 11.5, 0, '$+$'),
    op('addr', 12.6, 0, '$+$'),
    text('out', 13.5, 0, 'output', { w: 1.1 }),
  ],
  edges: [
    { from: 'in', to: 'd1', arrow: 'none' },
    { from: 'd1', to: 'ln' },
    { from: 'ln', to: 'sub' },
    { from: 'sub', to: 'dh', arrow: 'none' },
    { from: 'dh', to: 'down' },
    { from: 'down', to: 'f' },
    { from: 'f', to: 'up' },
    { from: 'up', to: 'adda' },
    { from: 'adda', to: 'addr' },
    { from: 'addr', to: 'out' },
    {
      from: 'dh:n',
      to: 'adda:n',
      via: [
        [5.3, -2.3],
        [11.5, -2.3],
      ],
    },
    {
      from: 'd1:s',
      to: 'addr:s',
      via: [
        [0.7, 2.2],
        [12.6, 2.2],
      ],
    },
  ],
  groups: [{ id: 'g', label: 'adapter', tone: 0, dashed: true, around: ['down', 'f', 'up', 'adda'], pad: 0.6 }],
}

/** Prompt tuning: trainable vectors prepended to the token embeddings at the input of the frozen stack. */
export const promptTuning: DiagramSpec = {
  unit: 40,
  nodes: [
    trainable('enc', 0, -0.72, 'prompt encoder\n(P-tuning only)', { w: 2.1, h: 0.9, small: true }),
    trainable('P', 2.7, -0.72, 'soft prompt $\\Pmat$', { w: 2, notes: { n: '$\\ell$ vectors of width $d$' } }),
    text('tok', 0, 0.54, '$x_1, \\dots, x_n$', { w: 1.6 }),
    frozen('emb', 2.7, 0.54, 'embedding', { w: 2 }),
    ...cells('c', 4.75, -1.08, 7, 3),
    anchor('cP', 4.18, -0.72),
    anchor('cT', 4.18, 0.54),
    anchor('cE', 5.32, 0),
    { id: 'stack', x: 7.2, y: 0, shape: 'stack', w: 2.2, h: 1.2, tone: 'neutral', label: '$L$ transformer\nblocks' },
    text('out', 9.3, 0, 'output', { w: 1.1 }),
    text('n', 4.75, 1.65, '$\\ell + n$ positions', { small: true, w: 2 }),
  ],
  edges: [
    { from: 'enc', to: 'P' },
    { from: 'P', to: 'cP:w' },
    { from: 'tok', to: 'emb' },
    { from: 'emb', to: 'cT:w' },
    { from: 'cE', to: 'stack' },
    { from: 'stack', to: 'out' },
  ],
  groups: [{ id: 'seq', around: ['c0', 'c6'], pad: 0.12 }],
}

/** Prefix tuning inside one attention head: trainable keys and values prepended to those of the context. */
export const prefixTuning: DiagramSpec = {
  unit: 40,
  nodes: [
    text('in', 0, 0.36, '$\\xvec$', { w: 0.6 }),
    dot('d', 0.7, 0.36),
    frozen('wq', 2.2, -1.9, '$\\Wmat_q$', { w: 0.9 }),
    frozen('wk', 2.2, 0.36, '$\\Wmat_k$', { w: 0.9 }),
    frozen('wv', 2.2, 3.36, '$\\Wmat_v$', { w: 0.9 }),
    ...cells('k', 4.75, -0.72, 5, 2),
    ...cells('v', 4.75, 2.28, 5, 2),
    text('pk', 3.5, -0.54, '$\\pvec_{1:\\ell}$', { tone: 0, w: 1 }),
    text('pv', 3.5, 2.46, '$\\uvec_{1:\\ell}$', { tone: 0, w: 1 }),
    anchor('kW', 4.18, 0.36),
    anchor('vW', 4.18, 3.36),
    anchor('kE', 5.32, 0),
    anchor('vE', 5.32, 3.0),
    frozen('sm', 6.8, 0, 'softmax', { w: 1.3 }),
    frozen('ws', 9.4, 0, 'weighted\nsum', { w: 1.3, h: 0.9 }),
    text('out', 11.0, 0, 'output', { w: 1.1 }),
  ],
  edges: [
    { from: 'in', to: 'd', arrow: 'none' },
    { from: 'd:n', to: 'wq:w', via: [[0.7, -1.9]] },
    { from: 'd', to: 'wk' },
    { from: 'd:s', to: 'wv:w', via: [[0.7, 3.36]] },
    { from: 'wq:e', to: 'sm:n', via: [[6.8, -1.9]], label: '$\\qvec$' },
    { from: 'wk', to: 'kW:w', label: '$\\kvec_{1:n}$' },
    { from: 'wv', to: 'vW:w', label: '$\\vvec_{1:n}$' },
    { from: 'kE', to: 'sm', label: 'keys' },
    { from: 'sm', to: 'ws', label: 'weights' },
    { from: 'vE:e', to: 'ws:s', via: [[9.4, 3.0]], label: 'values', labelPos: 0.3 },
    { from: 'ws', to: 'out' },
  ],
  groups: [
    { id: 'gk', around: ['k0', 'k4'], pad: 0.12 },
    { id: 'gv', around: ['v0', 'v4'], pad: 0.12 },
  ],
}

/** IA3: trainable vectors rescale the keys, the values and the feed-forward network's inner activation. */
export const ia3: DiagramSpec = {
  unit: 40,
  nodes: [
    text('in', 0, 0, '$\\xvec$', { w: 0.6 }),
    dot('d', 0.7, 0),
    frozen('wq', 2.0, -1.6, '$\\Wmat_q$', { w: 0.9 }),
    frozen('wk', 2.0, 0, '$\\Wmat_k$', { w: 0.9 }),
    frozen('wv', 2.0, 1.6, '$\\Wmat_v$', { w: 0.9 }),
    op('mk', 3.3, 0, '$\\odot$'),
    op('mv', 3.3, 1.6, '$\\odot$'),
    trainable('lk', 3.3, -0.8, '$\\lvec_k$', { w: 0.7, h: 0.5 }),
    trainable('lv', 3.3, 2.4, '$\\lvec_v$', { w: 0.7, h: 0.5 }),
    frozen('att', 5.0, 0, 'attention', { w: 1.5, h: 3.8 }),
    anchor('aq', 4.25, -1.6),
    anchor('ak', 4.25, 0),
    anchor('av', 4.25, 1.6),
    text('in2', 7.0, 0, '$\\xvec$', { w: 0.6 }),
    dot('d2', 7.7, 0),
    frozen('gate', 9.0, -1.0, 'gate'),
    frozen('up', 9.0, 1.0, 'up'),
    frozen('act', 10.4, -1.0, 'SiLU'),
    op('mul', 11.6, 0, '$\\odot$'),
    op('mff', 12.6, 0, '$\\odot$'),
    trainable('lff', 12.6, -0.85, '$\\lvec_{\\text{ff}}$', { w: 0.8, h: 0.5 }),
    frozen('down', 13.9, 0, 'down'),
    text('out', 15.1, 0, 'output', { w: 1.1 }),
  ],
  edges: [
    { from: 'in', to: 'd', arrow: 'none' },
    { from: 'd:n', to: 'wq:w', via: [[0.7, -1.6]] },
    { from: 'd', to: 'wk' },
    { from: 'd:s', to: 'wv:w', via: [[0.7, 1.6]] },
    { from: 'wq', to: 'aq:w' },
    { from: 'wk', to: 'mk' },
    { from: 'wv', to: 'mv' },
    { from: 'lk', to: 'mk' },
    { from: 'lv', to: 'mv' },
    { from: 'mk', to: 'ak:w' },
    { from: 'mv', to: 'av:w' },
    { from: 'in2', to: 'd2', arrow: 'none' },
    { from: 'd2:n', to: 'gate:w', via: [[7.7, -1.0]] },
    { from: 'd2:s', to: 'up:w', via: [[7.7, 1.0]] },
    { from: 'gate', to: 'act' },
    { from: 'act:e', to: 'mul:n', via: [[11.6, -1.0]] },
    { from: 'up:e', to: 'mul:s', via: [[11.6, 1.0]] },
    { from: 'mul', to: 'mff' },
    { from: 'lff', to: 'mff' },
    { from: 'mff', to: 'down' },
    { from: 'down', to: 'out' },
  ],
  groups: [
    { id: 'ga', label: 'attention', around: ['in', 'wq', 'wv', 'lk', 'lv', 'att'], pad: 0.3 },
    { id: 'gf', label: 'feed-forward', around: ['in2', 'gate', 'up', 'lff', 'down', 'out'], pad: 0.3 },
  ],
}

/** BitFit, sparse fine-tuning and partial fine-tuning train a chosen subset of the existing parameters. */
export const subsets: DiagramSpec = {
  unit: 40,
  nodes: [
    text('x', 0, 0, '$\\xvec$', { w: 0.6 }),
    frozen('W', 1.4, 0, '$\\Wmat$', { w: 0.9 }),
    op('add', 2.7, 0, '$+$'),
    trainable('b', 2.7, 1.0, '$\\bvec$', { w: 0.6, h: 0.5 }),
    text('y', 4.1, 0, '$\\Wmat\\xvec + \\bvec$', { w: 1.6 }),
    ...grid('w', 6.5, -0.72),
    text('wl', 7.22, 1.4, '$\\Wmat_0$', { w: 1 }),
    op('plus', 8.75, 0, '$+$'),
    ...grid('g', 9.5, -0.72, [
      [0, 2],
      [1, 0],
      [2, 3],
      [3, 1],
    ]),
    text('gl', 10.22, 1.4, '$\\deltavec$', { w: 1 }),
    frozen('emb', 0.8, 3.6, 'embeddings', { w: 1.6 }),
    frozen('b1', 2.9, 3.6, 'block 1', { w: 1.3 }),
    text('dots', 4.2, 3.6, '$\\cdots$', { w: 0.6 }),
    trainable('bl1', 5.6, 3.6, 'block $L - 1$', { w: 1.6 }),
    trainable('bl', 7.7, 3.6, 'block $L$', { w: 1.3 }),
    frozen('head', 9.7, 3.6, 'output head', { w: 1.7 }),
  ],
  edges: [
    { from: 'x', to: 'W' },
    { from: 'W', to: 'add' },
    { from: 'b', to: 'add' },
    { from: 'add', to: 'y' },
    { from: 'emb', to: 'b1' },
    { from: 'b1', to: 'dots' },
    { from: 'dots', to: 'bl1' },
    { from: 'bl1', to: 'bl' },
    { from: 'bl', to: 'head' },
  ],
  groups: [
    { id: 'g1', label: 'BitFit', around: ['x', 'W', 'b', 'y'], pad: 0.3 },
    { id: 'g2', label: 'sparse fine-tuning', around: ['w00', 'w33', 'wl', 'g00', 'g33', 'gl'], pad: 0.3 },
    { id: 'g3', label: 'partial fine-tuning', around: ['emb', 'b1', 'bl1', 'bl', 'head'], pad: 0.3 },
  ],
}

/** LoRA: a trainable low-rank path beside a frozen weight matrix. */
export const lora: DiagramSpec = {
  unit: 40,
  nodes: [
    text('in', 0, 0, '$\\xvec$', { w: 0.6 }),
    dot('d', 0.7, 0),
    frozen('W0', 3.3, -1.1, '$\\Wmat_0$', { w: 2.6, h: 0.8, notes: { n: '$d \\times k$' } }),
    {
      ...projector('encoder', 'A', 2.4, 1.1, '$\\Amat$', { w: 1.1, h: 1.3 }),
      highlight: true,
      notes: { s: '$r \\times k$' },
    },
    {
      ...projector('decoder', 'B', 4.1, 1.1, '$\\Bmat$', { w: 1.1, h: 1.3 }),
      highlight: true,
      notes: { s: '$d \\times r$' },
    },
    frozen('sc', 5.6, 1.1, '$\\times\\, \\alpha / r$', { w: 1 }),
    op('add', 6.9, 0, '$+$'),
    text('out', 7.7, 0, '$\\hvec$', { w: 0.6 }),
  ],
  edges: [
    { from: 'in', to: 'd', arrow: 'none' },
    { from: 'd:n', to: 'W0:w', via: [[0.7, -1.1]] },
    { from: 'd:s', to: 'A:w', via: [[0.7, 1.1]] },
    { from: 'W0:e', to: 'add:n', via: [[6.9, -1.1]] },
    { from: 'A', to: 'B' },
    { from: 'B', to: 'sc' },
    { from: 'sc:e', to: 'add:s', via: [[6.9, 1.1]] },
    { from: 'add', to: 'out' },
  ],
}

/** LoReFT: a low-rank edit of a hidden representation between two frozen blocks. */
export const reft: DiagramSpec = {
  unit: 40,
  nodes: [
    frozen('bl', 0.8, 0, 'block $\\ell$', { w: 1.5, h: 0.7 }),
    dot('dh', 2.3, 0, { label: '$\\hvec$', labelSide: 'n' }),
    trainable('wb', 4.3, 1.2, '$\\Wmat\\hvec + \\bvec$', { w: 1.7 }),
    trainable('rh', 4.3, 2.4, '$-\\Rmat\\hvec$', { w: 1.7 }),
    op('sum', 6.0, 1.8, '$+$'),
    trainable('rt', 7.2, 1.8, '$\\Rmat^{\\tr}$', { w: 0.9 }),
    op('add', 8.6, 0, '$+$'),
    frozen('bn', 10.2, 0, 'block $\\ell + 1$', { w: 1.7, h: 0.7 }),
  ],
  edges: [
    { from: 'bl', to: 'dh', arrow: 'none' },
    { from: 'dh', to: 'add' },
    { from: 'dh:s', to: 'wb:w', via: [[2.3, 1.2]] },
    { from: 'dh:s', to: 'rh:w', via: [[2.3, 2.4]] },
    { from: 'wb:e', to: 'sum:n', via: [[6.0, 1.2]] },
    { from: 'rh:e', to: 'sum:s', via: [[6.0, 2.4]] },
    { from: 'sum', to: 'rt' },
    { from: 'rt:e', to: 'add:s', via: [[8.6, 1.8]] },
    { from: 'add', to: 'bn' },
  ],
  groups: [
    {
      id: 'g',
      label: 'intervention',
      tone: 0,
      dashed: true,
      around: ['wb', 'rh', 'sum', 'rt'],
      labelAt: 'bottom-left',
    },
  ],
}
