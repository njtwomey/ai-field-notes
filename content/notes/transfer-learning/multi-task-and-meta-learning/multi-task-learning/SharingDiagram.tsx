import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const hard: DiagramSpec = {
  unit: 34,
  nodes: [
    { id: 'x', x: 0, y: 3, shape: 'circle', label: '$\\xvec$', tone: 'ink', filled: true },
    {
      id: 'shared',
      x: 2.8,
      y: 3,
      shape: 'stack',
      w: 2.2,
      h: 2.2,
      label: 'shared\nlayers\n$\\phivec(\\cdot;\\Wmat)$',
      tone: 0,
    },
    { id: 'h1', x: 6, y: 1.2, w: 1.8, h: 0.9, label: 'head 1', tone: 1 },
    { id: 'h2', x: 6, y: 3, w: 1.8, h: 0.9, label: 'head 2', tone: 2 },
    { id: 'h3', x: 6, y: 4.8, w: 1.8, h: 0.9, label: 'head 3', tone: 3 },
    { id: 'y1', x: 8.2, y: 1.2, shape: 'circle', w: 0.8, h: 0.8, label: '$y_1$', tone: 'ink' },
    { id: 'y2', x: 8.2, y: 3, shape: 'circle', w: 0.8, h: 0.8, label: '$y_2$', tone: 'ink' },
    { id: 'y3', x: 8.2, y: 4.8, shape: 'circle', w: 0.8, h: 0.8, label: '$y_3$', tone: 'ink' },
  ],
  edges: [
    { from: 'x', to: 'shared' },
    { from: 'shared:e', to: 'h1:w' },
    { from: 'shared:e', to: 'h2:w' },
    { from: 'shared:e', to: 'h3:w' },
    { from: 'h1', to: 'y1' },
    { from: 'h2', to: 'y2' },
    { from: 'h3', to: 'y3' },
  ],
}

const soft: DiagramSpec = {
  unit: 34,
  nodes: [
    { id: 'x1', x: 0, y: 1.2, shape: 'circle', w: 0.8, h: 0.8, label: '$\\xvec$', tone: 'ink', filled: true },
    { id: 'x2', x: 0, y: 3, shape: 'circle', w: 0.8, h: 0.8, label: '$\\xvec$', tone: 'ink', filled: true },
    { id: 'x3', x: 0, y: 4.8, shape: 'circle', w: 0.8, h: 0.8, label: '$\\xvec$', tone: 'ink', filled: true },
    { id: 'n1', x: 2.8, y: 1.2, w: 2.2, h: 0.9, label: 'network 1, $\\Wmat_1$', tone: 1 },
    { id: 'n2', x: 2.8, y: 3, w: 2.2, h: 0.9, label: 'network 2, $\\Wmat_2$', tone: 2 },
    { id: 'n3', x: 2.8, y: 4.8, w: 2.2, h: 0.9, label: 'network 3, $\\Wmat_3$', tone: 3 },
    { id: 'y1', x: 5.4, y: 1.2, shape: 'circle', w: 0.8, h: 0.8, label: '$y_1$', tone: 'ink' },
    { id: 'y2', x: 5.4, y: 3, shape: 'circle', w: 0.8, h: 0.8, label: '$y_2$', tone: 'ink' },
    { id: 'y3', x: 5.4, y: 4.8, shape: 'circle', w: 0.8, h: 0.8, label: '$y_3$', tone: 'ink' },
    {
      id: 'reg',
      x: 7.6,
      y: 3,
      w: 2.6,
      h: 1.3,
      label: 'coupling penalty\n$\\sum_t \\norm{\\Wmat_t - \\bar{\\Wmat}}^2$',
      tone: 'neutral',
      dashed: true,
    },
  ],
  edges: [
    { from: 'x1', to: 'n1' },
    { from: 'x2', to: 'n2' },
    { from: 'x3', to: 'n3' },
    { from: 'n1', to: 'y1' },
    { from: 'n2', to: 'y2' },
    { from: 'n3', to: 'y3' },
    {
      from: 'n1:n',
      to: 'reg:n',
      via: [
        [2.8, 0.2],
        [7.6, 0.2],
      ],
      dashed: true,
      arrow: 'none',
    },
    {
      from: 'n3:s',
      to: 'reg:s',
      via: [
        [2.8, 5.8],
        [7.6, 5.8],
      ],
      dashed: true,
      arrow: 'none',
    },
    { from: 'n2:e', to: 'reg:w', via: [[4.4, 3]], dashed: true, arrow: 'none' },
  ],
}

/** Hard and soft parameter sharing, side by side. */
export function SharingDiagram() {
  return (
    <Interactive
      title="Hard and soft parameter sharing"
      caption="Left: hard sharing. Every task uses the same hidden layers and has its own output head, so the shared weights receive gradients from every task. Right: soft sharing. Each task has its own network, and a penalty pulls the weights of the networks towards one another."
    >
      <div className="grid grid-cols-1 items-center gap-6 md:grid-cols-2">
        <Diagram spec={hard} ariaLabel="Hard sharing: input to shared layers, then three task-specific heads" />
        <Diagram
          spec={soft}
          ariaLabel="Soft sharing: three separate networks coupled by a penalty on the distance between their weights"
        />
      </div>
    </Interactive>
  )
}
