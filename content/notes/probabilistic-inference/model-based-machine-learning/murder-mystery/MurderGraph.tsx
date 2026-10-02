import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'right', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 46,
  nodes: [
    factor('prior', 2, 0, '$\\pr(m)$', 'e'),
    variable('m', 2, 1.5, '$m$'),
    factor('fw', 0.5, 3, '$\\pr(w \\mid m)$', 'w'),
    variable('w', 0.5, 4.5, '$w$', { filled: true }),
    factor('fh', 3.5, 3, '$\\pr(h \\mid m)$', 'e'),
    variable('h', 3.5, 4.5, '$h$', { filled: true }),
  ],
  edges: [
    line('prior', 'm', '$(0.3, 0.7)$'),
    line('fw', 'm', '$\\mu_{w \\to m} = (0.9, 0.2)$'),
    line('fh', 'm', '$\\mu_{h \\to m} = (0.5, 0.05)$'),
    line('fw', 'w'),
    line('fh', 'h'),
  ],
}

const STEPS: GraphStep[] = [
  { add: ['m'], text: 'The unknown is the murderer $m$, a variable with two states, Grey and Auburn.' },
  { add: ['prior'], text: 'Assumption 1, the prior: a factor $\\pr(m)$ with $\\pr(\\text{Grey}) = 0.3$.' },
  {
    add: ['fw', 'w'],
    text: 'Assumption 2: the weapon $w$ depends on the murderer through the factor $\\pr(w \\mid m)$. The revolver is observed, so $w$ is shaded.',
  },
  {
    add: ['fh', 'h'],
    text: 'Assumptions 3 and 4: the hair $h$ has its own factor $\\pr(h \\mid m)$ and no edge to $w$, so the clues are independent given $m$.',
  },
  {
    add: [],
    text: 'Inference: each observed clue sends $m$ the column of its table at the observed value (Grey, Auburn). The posterior is the normalised product of the three messages: $0.3 \\times 0.9 \\times 0.5$ against $0.7 \\times 0.2 \\times 0.05$, or 0.95 for Grey.',
  },
]

/** The murder-mystery factor graph, one assumption at a time, ending with the messages into the murderer node. */
export function MurderGraph() {
  return (
    <StepGraph
      title="Building the murder-mystery factor graph"
      caption="Step through the assumptions with the arrows. New nodes are highlighted; observed variables are shaded. The last step labels each edge into m with the message it carries, for a revolver and a grey hair."
      spec={SPEC}
      steps={STEPS}
      labelsFrom={4}
      ariaLabel="Factor graph of the murder mystery: a prior on the murderer and one factor for each clue"
    />
  )
}
