import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'Q', x: 4.4, y: 0.8, w: 3, h: 0.9, label: 'values $Q$ and policy', tone: 0 },
    { id: 'env', x: 0, y: 4, w: 2.6, h: 0.9, label: 'environment', tone: 'neutral' },
    { id: 'real', x: 4.4, y: 4, w: 3, h: 1, label: "real transition\n$(S, A, R, S')$", tone: 'neutral' },
    { id: 'model', x: 9.2, y: 4, w: 2.6, h: 1, label: 'model\n$\\hat p$, $\\hat r$', tone: 2 },
  ],
  edges: [
    { from: 'Q:w', to: 'env:n', via: [[0, 0.8]], label: 'act' },
    { from: 'env', to: 'real' },
    { from: 'real', to: 'Q', label: 'direct RL', labelRotate: false, labelSide: 'left' },
    { from: 'real', to: 'model', label: 'model learning' },
    {
      from: 'model:n',
      to: 'Q:e',
      via: [[9.2, 0.8]],
      label: 'planning: $n$ simulated updates',
      labelPos: 0.22,
      labelSide: 'left',
      labelRotate: false,
    },
  ],
}

/** Dyna: every real step feeds both a direct update and the model, and the model feeds n more updates. */
export function DynaDiagram() {
  return (
    <Interactive
      title="Dyna"
      caption="Each real transition is used twice: once for a Q-learning update, and once to update the model. The model then generates n simulated transitions, each given the same Q-learning update. Acting, learning and planning all improve the same value function."
    >
      <Diagram
        spec={spec}
        ariaLabel="Dyna: the agent acts in the environment; each real transition updates Q directly and updates a learned model; the model produces simulated transitions that also update Q"
      />
    </Interactive>
  )
}
