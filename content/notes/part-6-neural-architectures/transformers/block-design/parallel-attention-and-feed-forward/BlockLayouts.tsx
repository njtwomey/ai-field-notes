import { Diagram, Figure, op } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const box = (id: string, x: number, y: number, label: string, tone: number, w = 2.3): DiagramNode => ({
  id,
  x,
  y,
  label,
  tone,
  w,
  h: 0.6,
})

const spec: DiagramSpec = {
  unit: 34,
  nodes: [
    // Serial block, bottom to top.
    { id: 'st', x: 2, y: 9.4, shape: 'text', label: 'serial' },
    { id: 'sin', x: 2, y: 8.7, shape: 'text', label: '$\\xvec$' },
    { id: 'sd1', x: 2, y: 8.1, shape: 'dot' },
    box('sn1', 2, 7.3, 'norm', 2),
    box('sa', 2, 6.3, 'attention', 0),
    op('sp1', 2, 5.3, '$+$'),
    { id: 'sd2', x: 2, y: 4.7, shape: 'dot' },
    box('sn2', 2, 3.9, 'norm', 2),
    box('sf', 2, 2.9, 'feed-forward', 1),
    op('sp2', 2, 1.9, '$+$'),
    { id: 'sout', x: 2, y: 1.1, shape: 'text', label: '$\\yvec$' },
    // Parallel block.
    { id: 'pt', x: 8, y: 9.4, shape: 'text', label: 'parallel' },
    { id: 'pin', x: 8, y: 8.7, shape: 'text', label: '$\\xvec$' },
    { id: 'pd1', x: 8, y: 8.1, shape: 'dot' },
    box('pn', 8, 7.3, 'norm', 2),
    { id: 'pd2', x: 8, y: 6.6, shape: 'dot' },
    box('pa', 6.7, 5.4, 'attention', 0, 2.2),
    box('pf', 9.3, 5.4, 'feed-forward', 1, 2.2),
    op('pp', 8, 3.6, '$+$'),
    { id: 'pout', x: 8, y: 2.8, shape: 'text', label: '$\\yvec$' },
  ],
  edges: [
    { from: 'sin', to: 'sd1', arrow: 'none' },
    { from: 'sd1', to: 'sn1' },
    { from: 'sn1', to: 'sa' },
    { from: 'sa', to: 'sp1' },
    {
      from: 'sd1:w',
      to: 'sp1:w',
      via: [
        [0.4, 8.1],
        [0.4, 5.3],
      ],
    },
    { from: 'sp1', to: 'sd2', arrow: 'none' },
    { from: 'sd2', to: 'sn2' },
    { from: 'sn2', to: 'sf' },
    { from: 'sf', to: 'sp2' },
    {
      from: 'sd2:w',
      to: 'sp2:w',
      via: [
        [0.4, 4.7],
        [0.4, 1.9],
      ],
    },
    { from: 'sp2', to: 'sout' },
    { from: 'pin', to: 'pd1', arrow: 'none' },
    { from: 'pd1', to: 'pn' },
    { from: 'pn', to: 'pd2', arrow: 'none' },
    { from: 'pd2', to: 'pa:s', via: [[6.7, 6.6]] },
    { from: 'pd2', to: 'pf:s', via: [[9.3, 6.6]] },
    {
      from: 'pa:n',
      to: 'pp:s',
      via: [
        [6.7, 4.4],
        [8, 4.4],
      ],
    },
    {
      from: 'pf:n',
      to: 'pp:s',
      via: [
        [9.3, 4.4],
        [8, 4.4],
      ],
    },
    {
      from: 'pd1:w',
      to: 'pp:w',
      via: [
        [5.4, 8.1],
        [5.4, 3.6],
      ],
    },
    { from: 'pp', to: 'pout' },
  ],
}

/** Serial and parallel pre-norm blocks side by side. */
export function BlockLayouts() {
  return (
    <Figure
      title="Serial and parallel blocks"
      caption="Left: the usual pre-norm block, in which the feed-forward network reads the stream after attention has written to it. Right: the parallel block, in which both sublayers read the same normalised input and their outputs are added to the stream together. The parallel block has one sequential step per layer instead of two."
    >
      <Diagram spec={spec} ariaLabel="Serial and parallel transformer blocks" />
    </Figure>
  )
}
