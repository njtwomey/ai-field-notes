import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import { op } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 3.2, shape: 'text', w: 0.6, label: '$\\xvec_\\ell$' },
    { id: 'split', x: 0.9, y: 3.2, shape: 'dot' },
    { id: 'fc', x: 3, y: 3.2, shape: 'stack', w: 2.4, h: 1, label: '4 FC layers\nReLU', tone: 0 },
    { id: 'tb', x: 5.8, y: 4.1, w: 1, h: 0.6, label: '$\\thetavec^b_\\ell$', tone: 0 },
    { id: 'tf', x: 5.8, y: 2.3, w: 1, h: 0.6, label: '$\\thetavec^f_\\ell$', tone: 0 },
    { id: 'gb', x: 8.2, y: 4.1, w: 2.6, h: 0.7, label: 'backcast basis $\\vvec^b_i$', tone: 2 },
    { id: 'gf', x: 8.2, y: 2.3, w: 2.6, h: 0.7, label: 'forecast basis $\\vvec^f_i$', tone: 1 },
    op('sub', 11, 4.1, '$-$'),
    op('add', 11, 2.3, '$+$'),
    { id: 'xn', x: 13.6, y: 4.1, shape: 'text', w: 3.6, label: '$\\xvec_{\\ell+1}$ to the next block' },
    { id: 'yin', x: 11, y: 0.9, shape: 'text', w: 3.6, label: '$\\sum_{j<\\ell} \\hat\\yvec_j$ from earlier blocks' },
    { id: 'yout', x: 12.7, y: 2.3, shape: 'text', w: 1.6, label: '$\\sum_{j \\le \\ell} \\hat\\yvec_j$' },
  ],
  edges: [
    { from: 'x', to: 'split', arrow: 'none' },
    { from: 'split', to: 'fc' },
    {
      from: 'fc:e',
      to: 'tb:w',
      via: [
        [4.7, 3.2],
        [4.7, 4.1],
      ],
    },
    {
      from: 'fc:e',
      to: 'tf:w',
      via: [
        [4.7, 3.2],
        [4.7, 2.3],
      ],
    },
    { from: 'tb', to: 'gb' },
    { from: 'tf', to: 'gf' },
    { from: 'gb', to: 'sub', label: '$\\hat\\xvec_\\ell$' },
    { from: 'gf', to: 'add', label: '$\\hat\\yvec_\\ell$' },
    {
      from: 'split:n',
      to: 'sub:n',
      via: [
        [0.9, 5.4],
        [11, 5.4],
      ],
      label: 'input, carried past the block',
    },
    { from: 'yin', to: 'add' },
    { from: 'sub', to: 'xn' },
    { from: 'add', to: 'yout' },
  ],
  groups: [
    {
      id: 'blk',
      label: 'one block',
      tone: 0,
      around: ['fc', 'tb', 'tf', 'gb', 'gf'],
      pad: 0.35,
      labelAt: 'bottom-left',
    },
  ],
}

/** One N-BEATS block and its two residual branches. */
export function NBeatsBlockDiagram() {
  return (
    <Interactive
      title="An N-BEATS block and its two residual branches"
      caption="A fully connected stack produces two coefficient vectors. Each weights a set of basis vectors: one over the look-back window (the backcast) and one over the horizon (the partial forecast). The backcast is subtracted from the block's input before the next block sees it. The partial forecast is added to the running total."
    >
      <Diagram
        spec={spec}
        ariaLabel="N-BEATS block: input x through four fully connected layers to backcast and forecast coefficients, each weighting a basis; backcast subtracted from the input for the next block, forecast added to the running sum"
      />
    </Interactive>
  )
}
