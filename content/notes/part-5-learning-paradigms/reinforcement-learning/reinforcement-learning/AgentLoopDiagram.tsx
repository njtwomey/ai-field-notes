import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'val', x: 1, y: 1.2, w: 2.6, h: 1, label: 'value estimate\n$v_\\pi$ or $q_\\pi$', tone: 1 },
    { id: 'pol', x: 6, y: 1.2, w: 2.6, h: 1, label: 'policy\n$\\pi(a \\mid s)$', tone: 0 },
    { id: 'env', x: 3.5, y: 5, w: 3.2, h: 1, label: 'environment\nunknown MDP', tone: 'neutral' },
  ],
  edges: [
    { from: 'val', to: 'pol', route: 'curve', bend: -0.7, label: 'improvement', labelRotate: false },
    { from: 'pol', to: 'val', route: 'curve', bend: -0.7, label: 'evaluation', labelRotate: false },
    {
      from: 'pol:e',
      to: 'env:e',
      via: [
        [8.6, 1.2],
        [8.6, 5],
      ],
      label: 'action $A_t$',
      labelSide: 'right',
      labelRotate: false,
    },
    {
      from: 'env:w',
      to: 'val:w',
      via: [
        [-1.6, 5],
        [-1.6, 1.2],
      ],
      label: 'state $S_{t+1}$, reward $R_{t+1}$',
      labelSide: 'right',
      labelRotate: false,
    },
  ],
  groups: [{ id: 'agent', label: 'agent', tone: 0, around: ['val', 'pol'], pad: 0.6 }],
}

/** The agent–environment loop, with the agent's evaluation and improvement steps inside it. */
export function AgentLoopDiagram() {
  return (
    <Interactive
      title="The agent–environment loop"
      caption="The environment is known only through the transitions it returns. Inside the agent, evaluation estimates how good the current policy is and improvement moves the policy towards actions the estimate rates higher. Methods differ in which of the two they represent explicitly and how they estimate it from samples."
    >
      <Diagram
        spec={spec}
        ariaLabel="Agent-environment loop: policy sends action to the environment; environment returns state and reward to the value estimate; value and policy linked by evaluation and improvement"
      />
    </Interactive>
  )
}
