import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'env', x: 0, y: 1, w: 2.6, h: 0.9, label: 'environment', tone: 'neutral' },
    { id: 'Q', x: 5.4, y: 1, w: 3, h: 0.9, label: 'online network $Q_{\\thetavec}$', tone: 0 },
    { id: 'D', x: 0, y: 4, shape: 'stack', w: 2.6, h: 0.9, label: 'replay buffer $\\Dcal$', tone: 2 },
    { id: 'L', x: 5.4, y: 4, w: 3, h: 1.1, label: 'TD loss\n$\\paren{y - Q_{\\thetavec}(s, a)}^2$', tone: 'neutral' },
    { id: 'T', x: 5.4, y: 6.6, w: 3, h: 0.9, label: 'target network $Q_{\\thetavec^-}$', tone: 1 },
  ],
  edges: [
    { from: 'env', to: 'Q', label: 'state $s$' },
    {
      from: 'Q:n',
      to: 'env:n',
      via: [
        [5.4, -0.2],
        [0, -0.2],
      ],
      label: 'action $a$, $\\varepsilon$-greedy',
      labelSide: 'right',
    },
    { from: 'env', to: 'D', label: "store $(s, a, r, s')$" },
    { from: 'D', to: 'L', label: 'random minibatch' },
    {
      from: 'T',
      to: 'L',
      label: "$y = r + \\gamma \\max_{a'} Q_{\\thetavec^-}(s', a')$",
      labelRotate: false,
    },
    {
      from: 'L',
      to: 'Q',
      dashed: true,
      label: 'gradient step on $\\thetavec$',
      labelRotate: false,
      labelSide: 'right',
    },
    {
      from: 'Q:e',
      to: 'T:e',
      via: [
        [8.6, 1],
        [8.6, 6.6],
      ],
      dashed: true,
      label: 'copy $\\thetavec^- \\leftarrow \\thetavec$ every $C$ steps',
    },
  ],
}

/** Where DQN's two stabilisers sit: the buffer between acting and learning, and the frozen copy that sets targets. */
export function DqnDiagram() {
  return (
    <Interactive
      title="DQN: replay buffer and target network"
      caption="Acting and learning are decoupled. The online network acts in the environment and every transition goes into the replay buffer. Each gradient step regresses the online network on a random minibatch towards targets computed by the target network, a frozen copy that is refreshed every C steps."
    >
      <Diagram
        spec={spec}
        ariaLabel="DQN: environment and online Q-network exchange states and actions; transitions stored in a replay buffer; minibatches and targets from a target network feed a TD loss that updates the online network; online weights copied to the target network every C steps"
      />
    </Interactive>
  )
}
