import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 42,
  nodes: [
    factor('pp', 0, 0, '$\\Dir(\\alphavec)$', 'w'),
    variable('p', 0, 1.3, '$\\pvec$'),
    factor('pt', 0, 2.6, '$\\Cat$', 'w'),
    variable('t', 0, 3.9, '$t_i$'),
    { id: 'gdot', x: 1.9, y: 5.3, shape: 'dot', w: 0.18, h: 0.18 },
    factor('lab', 2.6, 5.3, '$\\Cat(\\pivec^{(k)}_c)$', 'n'),
    variable('l', 2.6, 7.0, '$\\ell_{ik}$', { filled: true }),
    variable('pi', 5.2, 5.3, '$\\pivec^{(k)}_c$'),
    factor('ppi', 6.8, 5.3, '$\\Dir(\\alphavec_c)$', 'e'),
    factor('smax', 5.2, 3.9, 'softmax', 'e'),
    variable('sk', 5.2, 2.6, '$\\svec^{(k)}_c$'),
    factor('gs', 5.2, 1.3, '$\\Gauss(\\svec^{(m_k)}_c, \\nu^{-1}\\Imat)$', 'e'),
    variable('sm', 5.2, 0, '$\\svec^{(m)}_c$'),
    variable('m', 3.2, 1.3, '$m_k$'),
    factor('ph', 3.2, 0, '$\\Cat(\\hvec)$', 'w'),
  ],
  edges: [
    line('pp', 'p'),
    line('p', 'pt'),
    line('pt', 't'),
    line('t', 'gdot', '$\\mu_{\\text{gate} \\to t_i}$'),
    line('lab', 'l'),
    link('pi', 'lab', false, { label: 'soft counts', labelSide: 'right', labelRotate: false, labelOffset: 0.3 }),
    line('ppi', 'pi'),
    line('smax', 'pi'),
    line('sk', 'smax'),
    line('gs', 'sk'),
    line('sm', 'gs'),
    line('m', 'gs'),
    line('ph', 'm'),
  ],
  groups: [
    { id: 'gate', label: '$t_i = c$', tone: 1, dashed: true, around: ['lab'], pad: 0.7, labelAt: 'bottom-left' },
    { id: 'I', label: 'items $i$', tone: 'ink', around: ['pt', 't', 'lab', 'l'], pad: 0.6, labelAt: 'bottom-right' },
    {
      id: 'K',
      label: 'workers $k$',
      tone: 'ink',
      around: ['lab', 'l', 'pi', 'smax', 'sk', 'm'],
      pad: 0.8,
      labelAt: 'bottom-right',
    },
    { id: 'M', label: 'communities $m$', tone: 'ink', around: ['sm'], pad: 0.35, labelAt: 'top-left' },
  ],
}

const STEPS: GraphStep[] = [
  {
    add: ['pp', 'p', 'pt', 't', 'I'],
    text: 'Each item has a true class $t_i$ drawn from class proportions $\\pvec$ with a Dirichlet prior.',
  },
  {
    add: ['lab', 'l', 'gate', 'gdot', 'pi', 'K'],
    text: "A worker's label is drawn from the row of their confusion matrix selected by the true class. The gate (dashed) holds one categorical factor per value $c$ of $t_i$; only the one with $t_i = c$ is on. Labels are observed.",
  },
  {
    add: ['ppi'],
    text: "BCC: every row of every worker's confusion matrix gets its own Dirichlet prior, favouring the diagonal.",
  },
  {
    add: ['smax', 'sk', 'gs', 'sm', 'm', 'ph', 'M'],
    remove: ['ppi'],
    text: "Community BCC: the Dirichlet prior is replaced. Each worker belongs to a community $m_k$; the worker's row scores are a Gaussian perturbation of the community's, and a softmax turns them into probabilities.",
  },
  {
    add: [],
    text: 'Messages: each label sends its item the probability of that label under each row, and sends each row soft counts weighted by the belief in $t_i$. The updates are written out below.',
  },
]

/** BCC and Community BCC as a factor graph, built step by step, with the gate that the true class switches. */
export function CrowdGraph() {
  return (
    <StepGraph
      title="From BCC to Community BCC"
      caption="Step through the model with the arrows. The dashed box is a gate selected by the true class. Labels are observed and shaded. Step 4 swaps BCC's Dirichlet prior on each worker's confusion rows for the community structure; the last step names the messages."
      spec={SPEC}
      steps={STEPS}
      labelsFrom={4}
      ariaLabel="Factor graph of Community BCC: true class with Dirichlet class proportions, a gated categorical factor for each label, worker confusion rows from softmax of scores perturbed around community scores"
    />
  )
}
