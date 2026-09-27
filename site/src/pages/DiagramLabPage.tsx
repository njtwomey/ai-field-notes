import { useState, type ReactNode } from 'react'
import { PageContainer } from '@/components/layout/AppShell'
import { Diagram } from '@/components/diagram/Diagram'
import { gate, merge, op, projector, reparam } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'

const lstm: DiagramSpec = {
  nodes: [
    { id: 'cprev', x: 0, y: 1, shape: 'text', label: '$\\cvec_{t-1}$' },
    { id: 'cout', x: 12.8, y: 1, shape: 'text', label: '$\\cvec_t$' },
    { id: 'hprev', x: 0, y: 5.4, shape: 'text', label: '$\\hvec_{t-1}$' },
    { id: 'hout', x: 12.8, y: 4, shape: 'text', label: '$\\hvec_t$' },
    { id: 'xt', x: 1.6, y: 6.9, shape: 'text', label: '$\\xvec_t$' },
    { id: 'bus', x: 1.6, y: 5.4, shape: 'dot' },
    op('fmul', 3, 1, '$\\otimes$'),
    op('add', 6.6, 1, '$\\oplus$'),
    gate('f', 3, 3.8, '$\\sigma$', 0),
    gate('i', 4.8, 3.8, '$\\sigma$', 0),
    gate('g', 6.6, 3.8, '$\\tanh$', 1),
    gate('o', 8.6, 3.8, '$\\sigma$', 0),
    { id: 'fl', x: 3, y: 4.55, shape: 'text', small: true, label: 'forget', tone: 'neutral' },
    { id: 'il', x: 4.8, y: 4.55, shape: 'text', small: true, label: 'input', tone: 'neutral' },
    { id: 'gl', x: 6.6, y: 4.55, shape: 'text', small: true, label: 'candidate', tone: 'neutral' },
    { id: 'ol', x: 8.6, y: 4.55, shape: 'text', small: true, label: 'output', tone: 'neutral' },
    op('imul', 6.6, 2.4, '$\\otimes$'),
    { id: 'ctanh', x: 10.4, y: 2.4, shape: 'pill', w: 0.9, h: 0.5, label: '$\\tanh$', tone: 1, small: true },
    op('omul', 10.4, 3.8, '$\\otimes$'),
  ],
  edges: [
    { from: 'cprev', to: 'fmul' },
    { from: 'fmul', to: 'add' },
    { from: 'add', to: 'cout' },
    { from: 'f:n', to: 'fmul:s' },
    { from: 'i:n', to: 'imul:w', via: [[4.8, 2.4]] },
    { from: 'g:n', to: 'imul:s' },
    { from: 'imul:n', to: 'add:s' },
    { from: 'add:e', to: 'ctanh:n', via: [[10.4, 1]] },
    { from: 'ctanh:s', to: 'omul:n' },
    { from: 'o:e', to: 'omul:w' },
    { from: 'omul:e', to: 'hout:w' },
    { from: 'hprev', to: 'bus', arrow: 'none' },
    { from: 'xt', to: 'bus', arrow: 'none' },
    { from: 'bus', to: 'f:s', via: [[3, 5.4]] },
    { from: 'bus', to: 'i:s', via: [[4.8, 5.4]] },
    { from: 'bus', to: 'g:s', via: [[6.6, 5.4]] },
    { from: 'bus', to: 'o:s', via: [[8.6, 5.4]] },
  ],
  groups: [
    {
      id: 'cell',
      label: 'LSTM cell',
      around: ['fmul', 'add', 'f', 'o', 'imul', 'ctanh', 'omul', 'bus', 'fl', 'ol'],
      pad: 0.45,
    },
  ],
}

const gru: DiagramSpec = {
  nodes: [
    { id: 'hprev', x: 0, y: 1, shape: 'text', label: '$\\hvec_{t-1}$' },
    { id: 'hout', x: 12.6, y: 1, shape: 'text', label: '$\\hvec_t$' },
    { id: 'xt', x: 1.6, y: 6.9, shape: 'text', label: '$\\xvec_t$' },
    { id: 'top', x: 1.6, y: 1, shape: 'dot' },
    { id: 'bus', x: 1.6, y: 5.4, shape: 'dot' },
    gate('r', 3.2, 3.8, '$\\sigma$', 0),
    gate('z', 5.2, 3.8, '$\\sigma$', 0),
    gate('c', 8, 3.8, '$\\tanh$', 1),
    { id: 'rl', x: 3.2, y: 4.55, shape: 'text', small: true, label: 'reset', tone: 'neutral' },
    { id: 'zl', x: 5.2, y: 4.55, shape: 'text', small: true, label: 'update', tone: 'neutral' },
    { id: 'cl', x: 8, y: 4.55, shape: 'text', small: true, label: 'candidate $\\tilde\\hvec_t$', tone: 'neutral' },
    op('rmul', 3.2, 2.4, '$\\otimes$'),
    { id: 'one', x: 6.6, y: 2.4, shape: 'pill', w: 0.8, h: 0.5, label: '$1-$', small: true },
    op('amul', 8, 1, '$\\otimes$'),
    op('bmul', 10.2, 2.4, '$\\otimes$'),
    op('add', 10.2, 1, '$\\oplus$'),
  ],
  edges: [
    { from: 'hprev', to: 'top', arrow: 'none' },
    { from: 'top', to: 'amul' },
    { from: 'amul', to: 'add' },
    { from: 'add', to: 'hout' },
    { from: 'top', to: 'bus', arrow: 'none' },
    { from: 'xt', to: 'bus', arrow: 'none' },
    { from: 'top:e', to: 'rmul:n', via: [[3.2, 1]] },
    { from: 'r:n', to: 'rmul:s' },
    {
      from: 'rmul:e',
      to: 'c:n',
      via: [
        [4.2, 2.4],
        [4.2, 3.1],
        [8, 3.1],
      ],
    },
    { from: 'z:n', to: 'one:w', via: [[5.2, 2.4]] },
    { from: 'one:e', to: 'amul:s', via: [[8, 2.4]] },
    {
      from: 'z:e',
      to: 'bmul:s',
      via: [
        [6.1, 3.8],
        [6.1, 3.3],
        [10.2, 3.3],
      ],
    },
    {
      from: 'c:e',
      to: 'bmul:e',
      via: [
        [11, 3.8],
        [11, 2.4],
      ],
    },
    { from: 'bmul:n', to: 'add:s' },
    { from: 'bus', to: 'r:s', via: [[3.2, 5.4]] },
    { from: 'bus', to: 'z:s', via: [[5.2, 5.4]] },
    { from: 'bus', to: 'c:s', via: [[8, 5.4]] },
  ],
  groups: [
    { id: 'cell', label: 'GRU cell', around: ['top', 'bus', 'r', 'rmul', 'amul', 'add', 'bmul', 'c', 'rl'], pad: 0.45 },
  ],
}

const block = (id: string, x: number, y: number, label: string, tone: number | 'neutral', h = 0.7) => ({
  id,
  x,
  y,
  label,
  tone,
  w: 2.8,
  h,
})

const transformer: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'ein', x: 3, y: 10.5, shape: 'text', label: 'inputs' },
    block('eemb', 3, 9.7, 'input embedding', 'neutral'),
    op('epe', 3, 8.8, '$\\oplus$'),
    { id: 'epel', x: 1, y: 8.8, shape: 'text', small: true, label: 'positional\nencoding' },
    { id: 'd1', x: 3, y: 8.25, shape: 'dot' },
    block('emha', 3, 7.4, 'multi-head attention', 0, 0.9),
    block('ean1', 3, 6.4, 'add & norm', 2, 0.5),
    { id: 'd2', x: 3, y: 5.9, shape: 'dot' },
    block('effn', 3, 5.2, 'feed-forward', 1),
    block('ean2', 3, 4.3, 'add & norm', 2, 0.5),
    { id: 'din', x: 8, y: 10.5, shape: 'text', label: 'outputs, shifted right' },
    block('demb', 8, 9.7, 'output embedding', 'neutral'),
    op('dpe', 8, 8.8, '$\\oplus$'),
    { id: 'dpel', x: 10, y: 8.8, shape: 'text', small: true, label: 'positional\nencoding' },
    { id: 'd3', x: 8, y: 8.25, shape: 'dot' },
    block('dmmha', 8, 7.4, 'masked multi-head\nattention', 0, 0.9),
    block('dan1', 8, 6.4, 'add & norm', 2, 0.5),
    { id: 'd4', x: 8, y: 5.9, shape: 'dot' },
    block('dxa', 8, 5.2, 'cross-attention', 3),
    block('dan2', 8, 4.3, 'add & norm', 2, 0.5),
    { id: 'd5', x: 8, y: 3.8, shape: 'dot' },
    block('dffn', 8, 3.1, 'feed-forward', 1),
    block('dan3', 8, 2.2, 'add & norm', 2, 0.5),
    block('lin', 8, 0.7, 'linear', 'neutral', 0.55),
    block('sm', 8, -0.15, 'softmax', 'neutral', 0.55),
    { id: 'dout', x: 8, y: -1, shape: 'text', label: 'output probabilities' },
  ],
  edges: [
    { from: 'ein', to: 'eemb' },
    { from: 'eemb', to: 'epe' },
    { from: 'epel', to: 'epe' },
    { from: 'epe', to: 'd1', arrow: 'none' },
    { from: 'd1', to: 'emha' },
    {
      from: 'd1:w',
      to: 'ean1:w',
      via: [
        [1.3, 8.25],
        [1.3, 6.4],
      ],
    },
    { from: 'emha', to: 'ean1' },
    { from: 'ean1', to: 'd2', arrow: 'none' },
    { from: 'd2', to: 'effn' },
    {
      from: 'd2:w',
      to: 'ean2:w',
      via: [
        [1.3, 5.9],
        [1.3, 4.3],
      ],
    },
    { from: 'effn', to: 'ean2' },
    {
      from: 'ean2:n',
      to: 'dxa:w',
      via: [
        [3, 3.5],
        [5.5, 3.5],
        [5.5, 5.2],
      ],
    },
    { from: 'din', to: 'demb' },
    { from: 'demb', to: 'dpe' },
    { from: 'dpel', to: 'dpe' },
    { from: 'dpe', to: 'd3', arrow: 'none' },
    { from: 'd3', to: 'dmmha' },
    {
      from: 'd3:e',
      to: 'dan1:e',
      via: [
        [9.7, 8.25],
        [9.7, 6.4],
      ],
    },
    { from: 'dmmha', to: 'dan1' },
    { from: 'dan1', to: 'd4', arrow: 'none' },
    { from: 'd4', to: 'dxa' },
    {
      from: 'd4:e',
      to: 'dan2:e',
      via: [
        [9.7, 5.9],
        [9.7, 4.3],
      ],
    },
    { from: 'dxa', to: 'dan2' },
    { from: 'dan2', to: 'd5', arrow: 'none' },
    { from: 'd5', to: 'dffn' },
    {
      from: 'd5:e',
      to: 'dan3:e',
      via: [
        [9.7, 3.8],
        [9.7, 2.2],
      ],
    },
    { from: 'dffn', to: 'dan3' },
    { from: 'dan3', to: 'lin' },
    { from: 'lin', to: 'sm' },
    { from: 'sm', to: 'dout' },
  ],
  groups: [
    { id: 'enc', label: 'encoder × $N$', tone: 0, around: ['emha', 'ean1', 'effn', 'ean2', 'd1', 'd2'], pad: 0.5 },
    {
      id: 'dec',
      label: 'decoder × $N$',
      tone: 1,
      around: ['dmmha', 'dan1', 'dxa', 'dan2', 'dffn', 'dan3', 'd3'],
      pad: 0.5,
    },
  ],
}

const autoencoder: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 2, w: 0.9, h: 0.9, label: '$\\xvec$' },
    projector('encoder', 'enc', 2.2, 2, 'encoder\n$f_{\\phivec}$'),
    { id: 'z', x: 4.4, y: 2, shape: 'latent', label: '$\\zvec$', tone: 2 },
    projector('decoder', 'dec', 6.6, 2, 'decoder\n$g_{\\thetavec}$', { tone: 1 }),
    { id: 'xh', x: 8.8, y: 2, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
  ],
  edges: [
    { from: 'x', to: 'enc' },
    { from: 'enc', to: 'z' },
    { from: 'z', to: 'dec' },
    { from: 'dec', to: 'xh' },
    {
      from: 'xh:s',
      to: 'x:s',
      via: [
        [8.8, 3.9],
        [0, 3.9],
      ],
      dashed: true,
      arrow: 'none',
      label: 'loss $\\norm{\\xvec - \\hat\\xvec}^2$',
    },
  ],
  groups: [{ id: 'bneck', label: 'bottleneck', tone: 2, dashed: true, around: ['z'], pad: 0.3 }],
}

const vae: DiagramSpec = merge(
  {
    nodes: [
      { id: 'x', x: 0, y: 3, w: 0.9, h: 0.9, label: '$\\xvec$' },
      projector('encoder', 'enc', 2.2, 3, 'encoder\n$q_{\\phivec}(\\zvec \\mid \\xvec)$', { w: 2.2, h: 3 }),
      { id: 'z', x: 9.2, y: 3, shape: 'latent', label: '$\\zvec$', tone: 2 },
      projector('decoder', 'dec', 11.5, 3, 'decoder\n$p_{\\thetavec}(\\xvec \\mid \\zvec)$', { w: 2.2, h: 3, tone: 1 }),
      { id: 'xh', x: 13.8, y: 3, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
      { id: 'kl', x: 4.8, y: 0.5, shape: 'text', small: true, label: 'KL term $\\KL(q_{\\phivec} \\,\\|\\, p)$' },
    ],
    edges: [
      { from: 'x', to: 'enc' },
      { from: 'rpadd', to: 'z' },
      { from: 'z', to: 'dec' },
      { from: 'dec', to: 'xh' },
      { from: 'rpmu:n', to: 'kl:s', dashed: true, arrow: 'none' },
    ],
  },
  reparam('rp', 4.8, 3, 'enc'),
)

const diffusion: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'x', x: 0, y: 1.5, w: 0.9, h: 0.9, label: '$\\xvec$' },
    projector('encoder', 'E', 2.1, 1.5, '$\\mathcal{E}$', { w: 1.4, h: 1.9 }),
    { id: 'z0', x: 4.3, y: 1.5, shape: 'latent', label: '$\\zvec_0$', tone: 0 },
    { id: 'dots', x: 6.6, y: 1.5, shape: 'text', label: '$\\cdots$' },
    { id: 'zt', x: 8.9, y: 1.5, shape: 'latent', label: '$\\zvec_t$', tone: 0 },
    { id: 'zT', x: 11.2, y: 1.5, shape: 'noise', label: '$\\zvec_T$', tone: 0 },
    {
      id: 'unet',
      x: 7.8,
      y: 4.1,
      w: 4,
      h: 0.9,
      label: 'denoiser $\\epsilonvec_{\\thetavec}(\\zvec_t, t, \\cvec)$',
      tone: 1,
    },
    { id: 'z0h', x: 4.3, y: 4.1, shape: 'latent', label: '$\\hat\\zvec_0$', tone: 0 },
    projector('decoder', 'D', 2.1, 4.1, '$\\mathcal{D}$', { w: 1.4, h: 1.9, dir: 'left', tone: 1 }),
    { id: 'xh', x: 0, y: 4.1, w: 0.9, h: 0.9, label: '$\\hat\\xvec$' },
    { id: 'cond', x: 7.8, y: 6.1, w: 2.8, h: 0.7, label: 'text encoder $\\tau_{\\thetavec}$', tone: 3 },
    { id: 'prompt', x: 7.8, y: 7.1, shape: 'text', label: 'prompt $y$' },
    {
      id: 'fwd',
      x: 9.2,
      y: 0.75,
      w: 6,
      shape: 'text',
      small: true,
      tone: 'neutral',
      label: 'forward: add noise, $q(\\zvec_t \\mid \\zvec_{t-1})$',
    },
    {
      id: 'rev',
      x: 7.8,
      y: 3.3,
      w: 6,
      shape: 'text',
      small: true,
      tone: 'neutral',
      label: 'reverse: $T$ denoising steps',
    },
  ],
  edges: [
    { from: 'x', to: 'E' },
    { from: 'E', to: 'z0' },
    { from: 'z0', to: 'dots', dashed: true },
    { from: 'dots', to: 'zt', dashed: true },
    { from: 'zt', to: 'zT', dashed: true },
    { from: 'zT:s', to: 'unet:e', via: [[11.2, 4.1]] },
    { from: 'unet', to: 'z0h' },
    { from: 'z0h', to: 'D' },
    { from: 'D', to: 'xh' },
    { from: 'prompt', to: 'cond' },
    { from: 'cond', to: 'unet', label: 'cross-attention' },
  ],
  groups: [
    { id: 'px', label: 'pixels', tone: 'neutral', around: ['x', 'xh'], pad: 0.4 },
    { id: 'lat', label: 'latent space', tone: 0, around: ['z0', 'zT', 'unet', 'z0h'], pad: 0.45 },
    { id: 'cg', label: 'conditioning', tone: 3, around: ['cond', 'prompt'], pad: 0.3, labelAt: 'bottom-right' },
  ],
}

const lda: DiagramSpec = {
  nodes: [
    { id: 'alpha', x: 0, y: 2.4, shape: 'circle', w: 0.7, h: 0.7, label: '$\\alpha$', tone: 'ink', small: true },
    { id: 'theta', x: 1.7, y: 2.4, shape: 'circle', label: '$\\thetavec_d$', tone: 'ink' },
    { id: 'z', x: 3.5, y: 2.4, shape: 'circle', label: '$z_{dn}$', tone: 'ink' },
    { id: 'w', x: 5.3, y: 2.4, shape: 'circle', filled: true, label: '$w_{dn}$', tone: 'ink' },
    { id: 'phi', x: 5.3, y: -0.7, shape: 'circle', label: '$\\phivec_k$', tone: 'ink' },
    { id: 'beta', x: 7.3, y: -0.7, shape: 'circle', w: 0.7, h: 0.7, label: '$\\beta$', tone: 'ink', small: true },
  ],
  edges: [
    { from: 'alpha', to: 'theta', route: 'straight' },
    { from: 'theta', to: 'z', route: 'straight' },
    { from: 'z', to: 'w', route: 'straight' },
    { from: 'phi', to: 'w', route: 'straight' },
    { from: 'beta', to: 'phi', route: 'straight' },
  ],
  groups: [
    { id: 'N', label: '$N_d$', tone: 'ink', around: ['z', 'w'], pad: 0.3, labelAt: 'bottom-right' },
    { id: 'D', label: '$D$', tone: 'ink', around: ['theta', 'z', 'w'], pad: 0.75, labelAt: 'bottom-right' },
    { id: 'K', label: '$K$', tone: 'ink', around: ['phi'], pad: 0.3, labelAt: 'bottom-right' },
  ],
}

const factorGraph: DiagramSpec = {
  nodes: [
    { id: 'a', x: 0, y: 2, shape: 'circle', label: '$x_1$', tone: 'ink' },
    { id: 'b', x: 2, y: 0, shape: 'circle', label: '$x_2$', tone: 'ink' },
    { id: 'c', x: 4, y: 2, shape: 'circle', label: '$x_3$', tone: 'ink', filled: true },
    { id: 'fab', x: 1, y: 1, shape: 'factor', label: '$\\psi_{12}$', labelSide: 'w' },
    { id: 'fbc', x: 3, y: 1, shape: 'factor', label: '$\\psi_{23}$', labelSide: 'e' },
    { id: 'fac', x: 2, y: 2, shape: 'factor', label: '$\\psi_{13}$', labelSide: 's' },
  ],
  edges: [
    { from: 'a', to: 'fab', route: 'straight', arrow: 'none' },
    { from: 'fab', to: 'b', route: 'straight', arrow: 'none' },
    { from: 'b', to: 'fbc', route: 'straight', arrow: 'none' },
    { from: 'fbc', to: 'c', route: 'straight', arrow: 'none' },
    { from: 'a', to: 'fac', route: 'straight', arrow: 'none' },
    { from: 'fac', to: 'c', route: 'straight', arrow: 'none' },
    { from: 'b', to: 'c', route: 'curve', bend: 1.1, dashed: true },
  ],
}

const ALARM: Record<string, { x: number; y: number; label: string }> = {
  B: { x: 0, y: 0, label: '$B$' },
  E: { x: 3, y: 0, label: '$E$' },
  A: { x: 1.5, y: 1.6, label: '$A$' },
  J: { x: 0, y: 3.2, label: '$J$' },
  M: { x: 3, y: 3.2, label: '$M$' },
}
const ALARM_EDGES: [string, string][] = [
  ['B', 'A'],
  ['E', 'A'],
  ['A', 'J'],
  ['A', 'M'],
]

/** Parents, children and the children's other parents. */
function markovBlanket(v: string): Set<string> {
  const parents = ALARM_EDGES.filter(([, t]) => t === v).map(([s]) => s)
  const children = ALARM_EDGES.filter(([s]) => s === v).map(([, t]) => t)
  const coParents = ALARM_EDGES.filter(([s, t]) => children.includes(t) && s !== v).map(([s]) => s)
  return new Set([...parents, ...children, ...coParents])
}

function MarkovBlanketDemo() {
  const [focus, setFocus] = useState('A')
  const blanket = markovBlanket(focus)
  const spec: DiagramSpec = {
    nodes: Object.entries(ALARM).map(([id, n]) => ({
      id,
      ...n,
      shape: 'circle' as const,
      tone: 'ink' as const,
      filled: blanket.has(id),
      highlight: id === focus,
    })),
    edges: ALARM_EDGES.map(([s, t]) => ({
      from: s,
      to: t,
      route: 'straight' as const,
      highlight: s === focus || t === focus,
    })),
  }
  return (
    <>
      <Diagram
        spec={spec}
        ariaLabel="Alarm network with the Markov blanket of the chosen node shaded"
        onNodeClick={setFocus}
      />
      <p className="mt-2 text-xs text-muted-foreground">
        Click a node. Its Markov blanket is shaded: {[...blanket].join(', ') || 'none'}.
      </p>
    </>
  )
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border p-5">
      <h2 className="mb-4 text-sm font-medium">{title}</h2>
      {children}
    </section>
  )
}

/** Test bench for the diagram library: architectures, projectors, the reparameterisation block and graphical models. */
export function DiagramLabPage() {
  return (
    <PageContainer>
      <h1 className="font-prose text-3xl font-bold">Diagram lab</h1>
      <p className="mt-2 mb-8 text-sm text-muted-foreground">
        Hand-specified diagrams: every node placed on a grid, groups drawn around nodes, edges routed through ports and
        waypoints, labels in KaTeX. Not linked from the site.
      </p>
      <div className="grid gap-6">
        <Panel title="LSTM cell">
          <Diagram spec={lstm} ariaLabel="LSTM cell" />
        </Panel>
        <Panel title="GRU cell">
          <Diagram spec={gru} ariaLabel="GRU cell" />
        </Panel>
        <Panel title="Transformer (Vaswani et al. 2017)">
          <Diagram spec={transformer} ariaLabel="Transformer encoder and decoder" />
        </Panel>
        <Panel title="Latent diffusion">
          <Diagram spec={diffusion} ariaLabel="Latent diffusion model" />
        </Panel>
        <Panel title="Autoencoder">
          <Diagram spec={autoencoder} ariaLabel="Autoencoder" />
        </Panel>
        <Panel title="Variational autoencoder with the reparameterisation block">
          <Diagram spec={vae} ariaLabel="Variational autoencoder" />
        </Panel>
        <div className="grid gap-6 md:grid-cols-3">
          <Panel title="LDA (plates)">
            <Diagram spec={lda} ariaLabel="Latent Dirichlet allocation plate diagram" />
          </Panel>
          <Panel title="Factor graph">
            <Diagram spec={factorGraph} ariaLabel="Factor graph on three variables" />
          </Panel>
          <Panel title="Markov blanket (clickable)">
            <MarkovBlanketDemo />
          </Panel>
        </div>
      </div>
    </PageContainer>
  )
}
