import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { op } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 2, shape: 'text', w: 0.6, label: '$\\xvec$' },
    { id: 'split', x: 1, y: 2, shape: 'dot' },
    { id: 'c1', x: 3, y: 2, w: 2.2, h: 1, label: '$3 \\times 3$ conv\nBN, ReLU', tone: 0 },
    { id: 'c2', x: 5.8, y: 2, w: 2.2, h: 1, label: '$3 \\times 3$ conv\nBN', tone: 0 },
    op('add', 7.9, 2, '$\\oplus$'),
    { id: 'relu', x: 9.2, y: 2, shape: 'pill', w: 1.1, h: 0.6, label: 'ReLU', tone: 'neutral', small: true },
    { id: 'y', x: 10.6, y: 2, shape: 'text', w: 0.6, label: '$\\yvec$' },
  ],
  edges: [
    { from: 'x', to: 'split', arrow: 'none' },
    { from: 'split', to: 'c1' },
    { from: 'c1', to: 'c2' },
    { from: 'c2', to: 'add' },
    { from: 'add', to: 'relu' },
    { from: 'relu', to: 'y' },
    {
      from: 'split:n',
      to: 'add:n',
      via: [
        [1, 0.2],
        [7.9, 0.2],
      ],
      label: 'shortcut: identity, or $\\Wmat_s \\xvec$ when the shape changes',
    },
  ],
  groups: [{ id: 'F', label: 'residual branch', tone: 0, around: ['c1', 'c2'], pad: 0.35, labelAt: 'bottom-left' }],
}

/** The basic residual block: two convolutions on one branch, the input carried unchanged on the other. */
export function ResidualBlockDiagram() {
  return (
    <Interactive
      title="A basic residual block"
      caption="The residual branch computes F(x) with two 3 × 3 convolutions; the shortcut carries x past them and the two are added before the final ReLU. With the branch's weights at zero the block is the identity."
    >
      <Diagram
        spec={spec}
        ariaLabel="Residual block: x splits into a shortcut and a branch of two convolutions; the branch output is added to x and passed through a ReLU to give y"
      />
    </Interactive>
  )
}
