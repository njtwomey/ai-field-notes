import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramNode, DiagramSpec } from '@/components/diagram/types'

const v = (id: string, x: number, y: number, extra: Partial<DiagramNode> = {}): DiagramNode => ({
  id,
  x,
  y,
  shape: 'circle',
  label: `$${id}$`,
  tone: 'ink',
  ...extra,
})
const note = (id: string, x: number, y: number, label: string, w = 3): DiagramNode => ({
  id,
  x,
  y,
  shape: 'text',
  small: true,
  w,
  label,
  tone: 'neutral',
})

const spec: DiagramSpec = {
  unit: 44,
  nodes: [
    v('X', 3, 0, { filled: true }),
    v('W', 0, 2),
    v('Y', 6, 2),
    v('M', 3, 2.9, { dashed: true }),
    v('C', 3, 5.6, { dashed: true }),
    note('xl', 3, -0.85, 'confounder: adjust'),
    note('ml', 3, 3.8, 'mediator: leave out'),
    note('cl', 3, 6.45, 'collider: leave out'),
    note('wl', -1.4, 2, 'treatment', 1.6),
    note('yl', 7.3, 2, 'outcome', 1.4),
  ],
  edges: [
    { from: 'X', to: 'W', route: 'straight', highlight: true },
    { from: 'X', to: 'Y', route: 'straight', highlight: true },
    { from: 'W', to: 'Y', route: 'straight' },
    { from: 'W', to: 'M', route: 'straight' },
    { from: 'M', to: 'Y', route: 'straight' },
    { from: 'W', to: 'C', route: 'straight' },
    { from: 'Y', to: 'C', route: 'straight' },
  ],
}

/** A causal graph with a confounder, a mediator and a collider. */
export function AdjustmentDag() {
  return (
    <Interactive
      title="What to adjust for"
      caption="The treatment W affects the outcome Y directly and through the mediator M. The confounder X causes both, opening the back-door path W ← X → Y (highlighted); adjusting for X (shaded) blocks it. Adjusting for M would remove part of the effect being estimated. The collider C is caused by both W and Y; the path through it is blocked until C is adjusted for, which opens it."
    >
      <Diagram
        spec={spec}
        ariaLabel="Causal graph: X points to W and Y; W points to Y, to the mediator M and to the collider C; M points to Y; Y points to C"
      />
    </Interactive>
  )
}
