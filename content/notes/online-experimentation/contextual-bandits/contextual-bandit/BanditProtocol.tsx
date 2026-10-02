import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'env', x: 0, y: 0, w: 2.8, h: 1, label: 'environment\n$x_t \\sim D$', tone: 'neutral' },
    { id: 'pol', x: 4.4, y: 0, w: 2.8, h: 1, label: 'policy\n$a_t \\sim \\pi(\\cdot \\mid x_t)$', tone: 0 },
    {
      id: 'rew',
      x: 8.8,
      y: 0,
      w: 3,
      h: 1,
      label: 'reward of chosen arm\n$r_t$, mean $\\mu(x_t, a_t)$',
      tone: 'neutral',
    },
    { id: 'upd', x: 4.4, y: 2.4, w: 2.8, h: 1, label: 'update estimates\nof $\\mu(x, a)$', tone: 0 },
    {
      id: 'log',
      x: 8.8,
      y: 2.4,
      w: 3,
      h: 1,
      label: 'log $(x_t, a_t, p_t, r_t)$\nfor off-policy evaluation',
      tone: 2,
      dashed: true,
    },
    { id: 'miss', x: 8.8, y: -1.25, shape: 'text', small: true, w: 3.6, label: 'rewards of other arms: never seen' },
  ],
  edges: [
    { from: 'env', to: 'pol', label: 'context' },
    { from: 'pol', to: 'rew', label: 'arm' },
    {
      from: 'rew:s',
      to: 'upd:e',
      via: [
        [8.8, 1.2],
        [6.4, 1.2],
        [6.4, 2.4],
      ],
    },
    { from: 'upd', to: 'pol', label: 'next round', labelSide: 'right' },
    { from: 'rew', to: 'log', dashed: true },
  ],
}

/** The interaction protocol of a contextual bandit. */
export function BanditProtocol() {
  return (
    <Interactive
      title="One round of a contextual bandit"
      caption="The environment reveals a context. The policy picks an arm, possibly at random with probability p_t, and only that arm's reward is observed. The learner updates its estimates before the next context arrives. Logging the context, arm, propensity p_t and reward makes the round reusable for evaluating other policies offline."
    >
      <Diagram
        spec={spec}
        ariaLabel="The environment draws a context; the policy chooses an arm; the chosen arm's reward is observed and used to update the policy; the round is logged with its propensity"
      />
    </Interactive>
  )
}
