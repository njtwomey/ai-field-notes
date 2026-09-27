import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 44,
  nodes: [
    factor('prior', 0, 0, '$\\Gauss(\\mu_i, \\sigma_i^2)$', 'n'),
    variable('w', 0, 1.5, '$w_i$'),
    factor('sum', 2.5, 1.5, '$\\Sigma$', 'n'),
    variable('s', 4.5, 1.5, '$s$'),
    factor('noise', 6.5, 1.5, '$\\Gauss(t; s, \\beta^2)$', 's'),
    variable('t', 8.5, 1.5, '$t$'),
    factor('gt', 10.5, 1.5, '$\\indicator[y t > 0]$', 's'),
    variable('y', 12.5, 1.5, '$y$', { filled: true }),
  ],
  edges: [
    line('prior', 'w'),
    line('w', 'sum', '$\\mu_{\\Sigma \\to w_i}$'),
    line('sum', 's'),
    line('s', 'noise'),
    line('noise', 't'),
    line('t', 'gt', '$\\mu_{> \\to t}$'),
    line('gt', 'y'),
  ],
  groups: [{ id: 'A', label: 'active $i$', tone: 'ink', around: ['prior', 'w'], pad: 0.45, labelAt: 'bottom-right' }],
}

const STEPS: GraphStep[] = [
  {
    add: ['prior', 'w', 'A'],
    text: 'Prior: every weight has its own Gaussian belief $\\Gauss(\\mu_i, \\sigma_i^2)$, independent of the others. The plate repeats over the active features of the impression.',
  },
  { add: ['sum', 's'], text: 'Score: $s$ is the sum of the active weights.' },
  { add: ['noise', 't'], text: 'Noise: $t \\sim \\Gauss(s, \\beta^2)$.' },
  { add: ['gt', 'y'], text: 'Threshold: the click $y = \\pm 1$ is the sign of $t$, and it is observed.' },
  {
    add: [],
    text: 'Messages: the threshold factor truncates $t$ to $y t > 0$; its message $\\mu_{> \\to t}$ is projected to a Gaussian by matching moments, and the change flows back through the sum as $\\mu_{\\Sigma \\to w_i}$, shared out in proportion to each $\\sigma_i^2$.',
  },
]

/** AdPredictor's factor graph for one impression, built layer by layer, with the two messages the update uses. */
export function AdPredictorGraph() {
  return (
    <StepGraph
      title="The AdPredictor factor graph for one impression"
      caption="Step through the four layers with the arrows. The click is observed and shaded. The last step labels the two messages the update is built from."
      spec={SPEC}
      steps={STEPS}
      labelsFrom={4}
      ariaLabel="Factor graph: Gaussian priors on the active weights, a sum to a score, Gaussian noise, and a threshold giving the click"
    />
  )
}
