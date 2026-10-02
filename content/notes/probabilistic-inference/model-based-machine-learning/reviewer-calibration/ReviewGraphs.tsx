import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string) => link(a, b, false)

const LINEAR: DiagramSpec = {
  unit: 44,
  nodes: [
    factor('pq', 0, 0, '$\\Gauss(0, \\alpha_q)$', 'w'),
    variable('q', 0, 1.4, '$q_p$'),
    factor('pb', 5, 0, '$\\Gauss(0, \\alpha_b)$', 'e'),
    variable('b', 5, 1.4, '$b_r$'),
    factor('sum', 2.5, 3, '$\\Gauss(y; \\mu + q + b, \\sigma^2)$', 'e'),
    variable('y', 2.5, 4.4, '$y_{rp}$', { filled: true }),
  ],
  edges: [line('pq', 'q'), line('pb', 'b'), line('q', 'sum'), line('b', 'sum'), line('sum', 'y')],
  groups: [
    { id: 'P', label: 'papers $p$', tone: 'ink', around: ['pq', 'q'], pad: 0.5, labelAt: 'bottom-right' },
    { id: 'R', label: 'reviewers $r$', tone: 'ink', around: ['pb', 'b'], pad: 0.5, labelAt: 'bottom-right' },
    { id: 'N', label: 'reviews $(r, p)$', tone: 'ink', around: ['sum', 'y'], pad: 0.5, labelAt: 'bottom-right' },
  ],
}

/** The linear-Gaussian calibration model as a factor graph. */
export function LinearReviewGraph() {
  return (
    <Interactive
      title="Factor graph of the linear calibration model"
      caption="Each paper has a quality and each reviewer a bias, each with a Gaussian prior. Every assigned review is one Gaussian factor joining its paper's quality and its reviewer's bias to the observed score. The review plate is indexed by the assigned pairs, so the graph's loops follow the assignment."
    >
      <Diagram
        spec={LINEAR}
        ariaLabel="Factor graph: paper quality and reviewer bias priors, joined by a Gaussian factor to each observed score"
      />
    </Interactive>
  )
}

const ORDINAL: DiagramSpec = {
  unit: 42,
  nodes: [
    factor('pq', 0, 0, '$\\Gauss$', 'w'),
    variable('q', 0, 1.4, '$q_p$'),
    factor('pl', 3, 0, '$\\GammaD$', 'e'),
    variable('lam', 3, 1.4, '$\\lambda_e$'),
    factor('score', 1.5, 3, '$\\Gauss(u; q_p, 1/\\lambda_e)$', 'w'),
    variable('u', 1.5, 4.4, '$u_n$'),
    factor('pa', 7, 0, '$\\GammaD$', 'e'),
    variable('a', 7, 1.4, '$a_r$'),
    factor('pt', 5.5, 2.4, '$\\Gauss(\\theta_\\ell, 1/a_r)$', 'e'),
    variable('th', 5.5, 3.7, '$\\theta_{r\\ell}$'),
    factor('obs', 3.5, 5.6, 'between thresholds', 'e'),
    variable('l', 3.5, 7, '$\\ell_n$', { filled: true }),
  ],
  edges: [
    line('pq', 'q'),
    line('pl', 'lam'),
    line('q', 'score'),
    line('lam', 'score'),
    line('score', 'u'),
    line('pa', 'a'),
    line('a', 'pt'),
    line('pt', 'th'),
    line('u', 'obs'),
    line('th', 'obs'),
    line('obs', 'l'),
  ],
  groups: [
    { id: 'P', label: 'papers $p$', tone: 'ink', around: ['pq', 'q'], pad: 0.45, labelAt: 'bottom-right' },
    { id: 'E', label: 'expertise $e$', tone: 'ink', around: ['pl', 'lam'], pad: 0.45, labelAt: 'bottom-right' },
    {
      id: 'R',
      label: 'reviewers $r$',
      tone: 'ink',
      around: ['pa', 'a', 'pt', 'th'],
      pad: 0.55,
      labelAt: 'bottom-right',
    },
    {
      id: 'N',
      label: 'reviews $n$',
      tone: 'ink',
      around: ['score', 'u', 'obs', 'l'],
      pad: 0.55,
      labelAt: 'bottom-right',
    },
  ],
}

const STEPS: GraphStep[] = [
  { add: ['pq', 'q', 'P'], text: 'Assumption 1: each submission has a latent quality $q_p$ with a Gaussian prior.' },
  {
    add: ['pl', 'lam', 'E', 'score', 'u', 'N'],
    text: 'Assumption 2: review $n$ has a latent score $u_n \\sim \\Gauss(q_p, 1/\\lambda_e)$, with a precision $\\lambda_e$ per self-declared expertise level and a Gamma prior on it.',
  },
  {
    add: ['pa', 'a', 'pt', 'th', 'R'],
    text: "Assumption 3: reviewer $r$ has personal thresholds $\\theta_{r\\ell}$, scattered around nominal ones with the reviewer's accuracy $a_r$ as precision.",
  },
  {
    add: ['obs', 'l'],
    text: 'Assumption 4: the observed recommendation $\\ell_n$ is the level whose thresholds bracket the latent score.',
  },
]

/** The ordinal reviewer model of SIGKDD'09, one assumption at a time. */
export function OrdinalReviewGraph() {
  return (
    <StepGraph
      title="Building the ordinal reviewer model"
      caption="Step through the assumptions with the arrows. Only the recommendations are observed. Each review belongs to one paper, one reviewer and one expertise level, so its factors sit in the review plate and reach across to the other plates."
      spec={ORDINAL}
      steps={STEPS}
      ariaLabel="Factor graph of the ordinal reviewer model: quality, expertise precision, reviewer thresholds and accuracy, and the observed recommendation"
    />
  )
}
