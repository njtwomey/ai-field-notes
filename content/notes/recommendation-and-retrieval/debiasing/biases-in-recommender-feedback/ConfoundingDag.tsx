import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'rel', x: 1, y: 1, shape: 'latent', w: 1.2, h: 1.2, label: '$R$' },
    { id: 'pol', x: 4, y: 1, shape: 'circle', w: 1.2, h: 1.2, label: '$\\pi_0$', tone: 1 },
    { id: 'pos', x: 7, y: 1, shape: 'circle', w: 1.2, h: 1.2, label: '$Z$', tone: 1 },
    { id: 'click', x: 4, y: 4, shape: 'circle', w: 1.2, h: 1.2, label: '$C$', filled: true },
    { id: 'pop', x: 7, y: 4, shape: 'circle', w: 1.2, h: 1.2, label: '$P$', filled: true },
    { id: 'rl', x: 1, y: 2.1, shape: 'text', small: true, label: 'true relevance' },
    { id: 'pl', x: 4, y: 0, shape: 'text', small: true, label: 'logging policy' },
    { id: 'zl', x: 7, y: 0, shape: 'text', small: true, label: 'exposure, position' },
    { id: 'cl', x: 4, y: 5.1, shape: 'text', small: true, label: 'click' },
    { id: 'ppl', x: 7, y: 5.1, shape: 'text', small: true, label: 'past popularity' },
  ],
  edges: [
    { from: 'rel', to: 'click', route: 'straight' },
    { from: 'rel', to: 'pol', route: 'straight', dashed: true, label: 'estimated' },
    { from: 'pol', to: 'pos', route: 'straight' },
    { from: 'pos', to: 'click', route: 'straight' },
    { from: 'pop', to: 'pol', route: 'straight' },
    { from: 'click', to: 'pop', route: 'straight', dashed: true, label: 'next round' },
  ],
}

/** Presentation as a confounder: the policy's view of relevance sets exposure, and exposure drives clicks. */
export function ConfoundingDag() {
  return (
    <Interactive
      title="Why logged clicks are confounded"
      caption="True relevance R causes clicks C. The logging policy π₀ decides exposure and position Z from its estimate of relevance and from past popularity P, and Z also causes clicks. A click is therefore correlated with anything that raised an item's position, not only with relevance. Clicks feed the next round's popularity, closing a feedback loop. Shaded nodes are observed; the double circle is latent."
    >
      <Diagram
        spec={spec}
        ariaLabel="Causal graph: relevance to click, relevance to policy, policy to position, position to click, popularity to policy, click to popularity"
      />
    </Interactive>
  )
}
