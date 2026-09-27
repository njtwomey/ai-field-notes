import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramGroup, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const v = (id: string, x: number, y: number, label: string, extra: Partial<DiagramNode> = {}): DiagramNode => ({
  id,
  x,
  y,
  shape: 'circle',
  label,
  tone: 'ink',
  ...extra,
})
const known = (id: string, x: number, y: number) =>
  v(id, x, y, '$\\sigma_j$', { w: 0.7, h: 0.7, small: true, filled: true })
const line = (from: string, to: string): DiagramEdge => ({ from, to, route: 'straight' })
const plate = (id: string, around: string[]): DiagramGroup => ({
  id,
  label: '$J$',
  tone: 'ink',
  around,
  pad: 0.3,
  labelAt: 'bottom-right',
})
const panel = (id: string, label: string, x: number): DiagramGroup => ({
  id,
  label,
  tone: 'neutral',
  rect: { x, y: -2.1, w: 3.6, h: 5.4 },
})

const spec: DiagramSpec = {
  nodes: [
    v('t1', 2, 0.3, '$\\theta_j$'),
    v('y1', 2, 1.9, '$y_j$', { filled: true }),
    known('s1', 0.9, 1.9),
    v('t2', 6.6, -0.8, '$\\theta$'),
    v('y2', 6.6, 1.9, '$y_j$', { filled: true }),
    known('s2', 5.5, 1.9),
    v('mu', 10.4, -1.1, '$\\mu$', { w: 0.75, h: 0.75 }),
    v('tau', 11.9, -1.1, '$\\tau$', { w: 0.75, h: 0.75 }),
    v('t3', 11.2, 0.3, '$\\theta_j$'),
    v('y3', 11.2, 1.9, '$y_j$', { filled: true }),
    known('s3', 10.1, 1.9),
  ],
  edges: [
    line('t1', 'y1'),
    line('s1', 'y1'),
    line('t2', 'y2'),
    line('s2', 'y2'),
    line('mu', 't3'),
    line('tau', 't3'),
    line('t3', 'y3'),
    line('s3', 'y3'),
  ],
  groups: [
    panel('p1', 'no pooling', 0),
    panel('p2', 'complete pooling', 4.6),
    panel('p3', 'partial pooling', 9.2),
    plate('j1', ['t1', 'y1', 's1']),
    plate('j2', ['y2', 's2']),
    plate('j3', ['t3', 'y3', 's3']),
  ],
}

/** Three ways to model J related groups, as plate diagrams. */
export function PoolingPlates() {
  return (
    <Interactive
      title="No pooling, complete pooling and partial pooling"
      caption="Each group j has an observed estimate y_j (shaded) with known standard error σ_j. No pooling gives every group its own unrelated θ_j. Complete pooling uses one θ for all groups, outside the plate. Partial pooling, the hierarchical model, draws each θ_j from a population with mean μ and spread τ, which are themselves estimated from all J groups."
    >
      <Diagram
        spec={spec}
        ariaLabel="Three plate diagrams: separate theta_j per group; one shared theta; and theta_j drawn from a population with hyperparameters mu and tau"
      />
    </Interactive>
  )
}
