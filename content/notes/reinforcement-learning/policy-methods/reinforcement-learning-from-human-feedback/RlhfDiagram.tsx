import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  spread: [1, 1.15],
  nodes: [
    { id: 'pre', x: 0, y: 1, w: 2.6, h: 0.9, label: 'pretrained model', tone: 'neutral' },
    { id: 'sft', x: 3.8, y: 1, w: 3, h: 1, label: 'supervised fine-tuning\non demonstrations', tone: 'neutral' },
    { id: 'ref', x: 7.6, y: 1, w: 1.8, h: 0.8, label: '$\\pi_{\\text{ref}}$', tone: 0 },
    { id: 'pol', x: 7.6, y: 3.6, w: 1.8, h: 0.8, label: '$\\pi_{\\thetavec}$', tone: 0 },
    {
      id: 'rew',
      x: 13,
      y: 3.6,
      w: 3.8,
      h: 1.1,
      label:
        'penalised reward\n$r_\\phi(x, y) - \\beta \\log \\frac{\\pi_{\\thetavec}(y \\mid x)}{\\pi_{\\text{ref}}(y \\mid x)}$',
      tone: 'neutral',
    },
    { id: 'cmp', x: 0, y: 6.4, w: 2.6, h: 1, label: 'human comparisons\n$(x, y_w, y_l)$', tone: 'neutral' },
    { id: 'rm', x: 3.8, y: 6.4, w: 3, h: 1, label: 'reward model $r_\\phi$\nBradley–Terry loss', tone: 2 },
  ],
  edges: [
    { from: 'pre', to: 'sft' },
    { from: 'sft', to: 'ref' },
    { from: 'ref', to: 'pol', label: 'initialise', labelRotate: false, labelSide: 'right' },
    { from: 'pol', to: 'rew', label: 'response $y$' },
    {
      from: 'ref:e',
      to: 'rew:n',
      via: [[13, 1]],
      dashed: true,
      label: 'reference for the KL term',
    },
    { from: 'cmp', to: 'rm' },
    {
      from: 'rm:e',
      to: 'rew:e',
      via: [
        [15.6, 6.4],
        [15.6, 3.6],
      ],
      label: 'score $r_\\phi(x, y)$',
    },
    {
      from: 'rew:s',
      to: 'pol:w',
      via: [
        [13, 5],
        [5.9, 5],
        [5.9, 3.6],
      ],
      dashed: true,
      label: 'PPO update of $\\thetavec$',
    },
  ],
  groups: [
    { id: 'g1', label: 'step 1: supervised policy', tone: 'neutral', around: ['pre', 'ref'], pad: 0.3 },
    { id: 'g2', label: 'step 2: reward model', tone: 2, around: ['cmp', 'rm'], pad: 0.3, labelAt: 'bottom-left' },
    {
      id: 'g3',
      label: 'step 3: RL against the reward model',
      tone: 0,
      around: ['pol', 'rew'],
      pad: 0.35,
      labelAt: 'bottom-left',
    },
  ],
}

/** The three stages of RLHF and what flows between them. */
export function RlhfDiagram() {
  return (
    <Interactive
      title="The RLHF pipeline"
      caption="A supervised policy is trained first and frozen as the reference. A reward model is fitted to human comparisons of pairs of responses. The policy, initialised from the reference, is then optimised by PPO against the reward model's score minus a KL penalty that keeps it close to the reference."
    >
      <Diagram
        spec={spec}
        ariaLabel="RLHF pipeline: pretrained model fine-tuned into a reference policy; human comparisons train a reward model; the policy generates responses scored by the reward model minus a KL penalty to the reference, and is updated by PPO"
      />
    </Interactive>
  )
}
