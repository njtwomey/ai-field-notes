import { Diagram, Figure, op } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'actor', x: 4, y: 0.8, w: 2.6, h: 1, label: 'actor\n$\\pi_{\\thetavec}(a \\mid s)$', tone: 0 },
    { id: 'critic', x: 4, y: 3.2, w: 2.6, h: 1, label: 'critic\n$V_{\\wvec}(s)$', tone: 1 },
    op('td', 7, 3.2, '$\\delta$'),
    { id: 'env', x: 5, y: 5.8, w: 2.8, h: 0.9, label: 'environment', tone: 'neutral' },
    { id: 'j', x: 0.6, y: 3.2, shape: 'dot' },
  ],
  edges: [
    {
      from: 'actor:e',
      to: 'env:e',
      via: [
        [9, 0.8],
        [9, 5.8],
      ],
      label: 'action $A_t$',
    },
    { from: 'env:w', to: 'j', via: [[0.6, 5.8]], arrow: 'none', label: '$S_{t+1}$, $R_{t+1}$', labelSide: 'right' },
    { from: 'j', to: 'actor:w', via: [[0.6, 0.8]] },
    { from: 'j', to: 'critic:w' },
    { from: 'critic', to: 'td' },
    {
      from: 'td:n',
      to: 'actor:s',
      via: [
        [7, 2],
        [4, 2],
      ],
      label: 'TD error $\\delta_t$',
      labelSide: 'right',
    },
    {
      from: 'td:s',
      to: 'critic:s',
      via: [
        [7, 4.4],
        [4, 4.4],
      ],
      dashed: true,
      label: 'TD update of $\\wvec$',
    },
  ],
}

/** The critic turns each transition into a TD error, which both trains the critic and scores the actor's action. */
export function ActorCriticDiagram() {
  return (
    <Figure
      title="One-step actor-critic"
      caption="The actor chooses the action. The critic values states, and after each transition forms the TD error δ = R + γV(S′) − V(S). The same δ updates the critic (dashed) and serves as the actor's advantage estimate: actions followed by a positive δ become more likely."
    >
      <Diagram
        spec={spec}
        ariaLabel="Actor-critic: the actor sends action A_t to the environment; the next state and reward go to actor and critic; the critic's TD error updates the critic and the actor"
      />
    </Figure>
  )
}
