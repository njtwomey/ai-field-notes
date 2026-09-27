import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const stage = (id: string, x: number, label: string, tone: number, h = 1.3) => ({
  id,
  x,
  y: 2,
  w: 2.6,
  h,
  label,
  tone,
})

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'cat', x: 0.6, y: 2, shape: 'stack', w: 2.2, h: 1.6, label: 'catalogue\n$10^6$ to $10^9$', tone: 'neutral' },
    stage('ret', 4.2, 'retrieval\ntwo-tower + ANN', 0, 1.6),
    stage('rank', 7.8, 'ranking\ncross features', 1, 1.2),
    stage('rerank', 11.4, 're-ranking\ndiversity, rules', 2, 0.9),
    { id: 'slate', x: 14.6, y: 2, shape: 'pill', w: 1.8, h: 0.8, label: 'slate', tone: 'ink' },
    { id: 'n1', x: 5.9, y: 0.7, shape: 'text', small: true, label: '$10^2$ to $10^3$' },
    { id: 'n2', x: 9.5, y: 0.7, shape: 'text', small: true, label: '$10^1$ to $10^2$' },
    { id: 'n3', x: 13.05, y: 0.7, shape: 'text', small: true, label: '$\\approx 10$' },
    { id: 'm1', x: 4.2, y: 3.4, shape: 'text', small: true, label: 'recall, speed' },
    { id: 'm2', x: 7.8, y: 3.4, shape: 'text', small: true, label: 'precision' },
    { id: 'm3', x: 11.4, y: 3.4, shape: 'text', small: true, label: 'whole-list goals' },
    { id: 'user', x: 14.6, y: 4.6, shape: 'circle', w: 1, h: 1, label: 'user' },
    { id: 'log', x: 7.8, y: 4.6, w: 2.8, h: 0.8, label: 'logged feedback', tone: 'neutral', dashed: true },
  ],
  edges: [
    { from: 'cat', to: 'ret' },
    { from: 'ret', to: 'rank' },
    { from: 'rank', to: 'rerank' },
    { from: 'rerank', to: 'slate' },
    { from: 'slate:s', to: 'user:n' },
    { from: 'user:w', to: 'log:e', label: 'clicks, views' },
    { from: 'log:w', to: 'ret:s', via: [[4.2, 4.6]], dashed: true, label: 'retrain' },
  ],
}

/** The retrieve, rank, re-rank funnel and the feedback loop that closes it. */
export function Funnel() {
  return (
    <Interactive
      title="The multi-stage funnel"
      caption="Each stage sees fewer items and can afford more computation per item. Retrieval uses models whose item side can be precomputed; ranking uses models that combine user and item features; re-ranking scores the list as a whole. What the user does with the slate is logged and becomes the next model's training data."
    >
      <Diagram
        spec={spec}
        ariaLabel="Catalogue to retrieval to ranking to re-ranking to slate, with logged feedback feeding retraining"
      />
    </Interactive>
  )
}
