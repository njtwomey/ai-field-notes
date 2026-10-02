import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'uf', x: 2.4, y: 8, w: 3.6, h: 0.7, label: 'user and context features', tone: 'neutral' },
    { id: 'if', x: 8.4, y: 8, w: 3.6, h: 0.7, label: 'item features', tone: 'neutral' },
    { id: 'ut', x: 2.4, y: 6, shape: 'stack', w: 2.6, h: 1.4, label: 'query tower', tone: 0 },
    { id: 'it', x: 8.4, y: 6, shape: 'stack', w: 2.6, h: 1.4, label: 'item tower', tone: 1 },
    { id: 'u', x: 2.4, y: 4, w: 1.4, h: 0.6, label: '$\\uvec$', tone: 0 },
    { id: 'v', x: 8.4, y: 4, w: 1.4, h: 0.6, label: '$\\vvec$', tone: 1 },
    { id: 'dot', x: 5.4, y: 2.4, shape: 'op', label: '$\\cdot$' },
    { id: 's', x: 5.4, y: 1.2, shape: 'text', label: 'score $\\uvec\\transpose\\vvec$' },
    { id: 'idx', x: 11.8, y: 4, w: 2.6, h: 0.9, label: 'ANN index\nof all $\\vvec$', tone: 'neutral', dashed: true },
    { id: 'off', x: 11.8, y: 6, shape: 'text', small: true, label: 'offline:\nevery item' },
    { id: 'on', x: 0.2, y: 6, shape: 'text', small: true, label: 'online:\nper request' },
  ],
  edges: [
    { from: 'uf', to: 'ut' },
    { from: 'if', to: 'it' },
    { from: 'ut', to: 'u' },
    { from: 'it', to: 'v' },
    { from: 'u:s', to: 'dot:w', via: [[2.4, 2.4]] },
    { from: 'v:s', to: 'dot:e', via: [[8.4, 2.4]] },
    { from: 'v:e', to: 'idx:w', dashed: true },
  ],
}

/** Two towers that meet only at a dot product, which is what lets the item side be indexed offline. */
export function TwoTowerDiagram() {
  return (
    <Interactive
      title="The two-tower model"
      caption="User and item features never meet until the final dot product. The item tower is run offline over the whole catalogue and its outputs are stored in an approximate nearest-neighbour index; at request time only the query tower runs, and its vector retrieves the items with the largest dot products."
    >
      <Diagram
        spec={spec}
        ariaLabel="Two-tower model: user features to query tower to u, item features to item tower to v, dot product score, item vectors in an ANN index"
      />
    </Interactive>
  )
}
