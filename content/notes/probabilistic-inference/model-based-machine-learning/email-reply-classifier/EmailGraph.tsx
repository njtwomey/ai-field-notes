import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string) => link(a, b, false)

const SPEC: DiagramSpec = {
  unit: 40,
  nodes: [
    factor('pw', -1.3, -0.8, '$\\Gauss(0, 1)$', 'n'),
    variable('w', 0, -0.8, '$w_i$'),
    variable('x', 0, 1.6, '$x_{ni}$', { filled: true }),
    factor('mul', 1.4, 0.8, '$\\times$', 'n'),
    variable('p', 2.8, 0.8, '$z_{ni}$', { w: 0.9, h: 0.9 }),
    factor('sum', 4.4, 0.8, '$\\Sigma$', 'n'),
    variable('s', 5.8, 0.8, '$s_n$'),
    factor('noise', 7.2, 0.8, '$\\Gauss(s_n, \\sigma^2)$', 's'),
    variable('sn', 8.6, 0.8, '$\\tilde s_n$'),
    factor('pt', 8.6, -1.6, '$\\Gauss(0, 10)$', 'e'),
    variable('t', 10, -1.6, '$\\theta$'),
    factor('gt', 10, 0.8, '$>$', 'n'),
    variable('r', 11.4, 0.8, '$r_n$', { filled: true }),
  ],
  edges: [
    line('pw', 'w'),
    line('w', 'mul'),
    line('x', 'mul'),
    line('mul', 'p'),
    line('p', 'sum'),
    line('sum', 's'),
    line('s', 'noise'),
    line('noise', 'sn'),
    line('sn', 'gt'),
    line('pt', 't'),
    line('t', 'gt'),
    line('gt', 'r'),
  ],
  groups: [
    { id: 'I', label: 'features $i$', tone: 'ink', around: ['w', 'x', 'mul', 'p'], pad: 0.35, labelAt: 'bottom-right' },
    {
      id: 'N',
      label: 'emails $n$',
      tone: 'ink',
      around: ['x', 'mul', 'p', 'sum', 's', 'noise', 'sn', 'gt', 'r'],
      pad: 0.75,
      labelAt: 'bottom-right',
    },
  ],
}

const STEPS: GraphStep[] = [
  {
    add: ['x', 'I', 'N'],
    text: 'Assumption 1: every feature value $x_{ni}$ can be computed for every email, so the features are observed. Plates repeat over features $i$ and emails $n$.',
  },
  {
    add: ['mul', 'p', 'sum', 's', 'w'],
    text: 'Assumptions 2 and 3: each email has a continuous score $s_n = \\sum_i w_i x_{ni}$, with one weight per feature shared by every email.',
  },
  {
    add: ['pw'],
    text: 'Assumptions 4 and 5: a weight is as likely to be positive as negative and usually small, so $w_i \\sim \\Gauss(0, 1)$.',
  },
  {
    add: ['noise', 'sn'],
    text: 'Gaussian noise of variance $\\sigma^2 = 10$ turns the score into a noisy score, so no single email gets probability zero.',
  },
  {
    add: ['pt', 't', 'gt', 'r'],
    text: 'The user replies when the noisy score exceeds a learned threshold $\\theta \\sim \\Gauss(0, 10)$. The reply $r_n$ is observed during training.',
  },
]

/** The reply classifier as a factor graph, built one assumption at a time. */
export function EmailGraph() {
  return (
    <StepGraph
      title="Building the reply classifier"
      caption="Step through the assumptions with the arrows. Each feature bucket i has a weight shared by every email; email n multiplies its feature values by the weights, sums them into a score, adds noise and compares the result with a shared threshold. Shaded nodes are observed."
      spec={SPEC}
      steps={STEPS}
      ariaLabel="Factor graph: weight and feature value multiply, sum to a score, noise gives a noisy score, compared with a threshold to give the reply"
    />
  )
}
