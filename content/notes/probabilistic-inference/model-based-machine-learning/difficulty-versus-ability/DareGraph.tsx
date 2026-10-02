import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 40,
  nodes: [
    factor('pa', 0, 0, '$\\Gauss(\\mu_p, \\sigma_p^2)$', 'e'),
    variable('a', 0, 1.3, '$a_p$'),
    factor('pd', 9.2, 0, '$\\Gauss(\\mu_q, \\sigma_q^2)$', 'w'),
    variable('d', 9.2, 1.3, '$d_q$'),
    factor('pt', 10.8, 0, '$\\GammaD$', 'e'),
    variable('tau', 10.8, 1.3, '$\\tau_q$'),
    factor('sub', 2.8, 2.4, '$a - d$', 'n'),
    variable('t', 2.8, 3.7, '$t_{pq}$'),
    factor('know', 4.6, 3.7, '$\\Phi(\\sqrt{\\tau_q}\\, t_{pq})$', 's'),
    variable('c', 6.4, 3.7, '$c_{pq}$'),
    { id: 'dF', x: 5.8, y: 5.0, shape: 'dot', w: 0.18, h: 0.18 },
    { id: 'dT', x: 7.0, y: 5.0, shape: 'dot', w: 0.18, h: 0.18 },
    factor('uni', 5.2, 5.7, 'uniform', 'w'),
    factor('eq', 7.6, 5.7, '$r = y_q$', 'n'),
    variable('r', 6.4, 7.1, '$r_{pq}$', { filled: true }),
    variable('y', 9.8, 5.7, '$y_q$'),
    factor('py', 9.8, 7.1, 'uniform', 'e'),
  ],
  edges: [
    line('pa', 'a'),
    line('pd', 'd'),
    line('pt', 'tau'),
    line('a', 'sub'),
    line('d', 'sub'),
    line('sub', 't'),
    line('t', 'know'),
    line('tau', 'know'),
    line('know', 'c'),
    line('c', 'dF'),
    line('c', 'dT'),
    line('eq', 'r'),
    line('uni', 'r'),
    line('y', 'eq', '$\\mu_{r \\to y_q}$'),
    line('py', 'y'),
  ],
  groups: [
    {
      id: 'gT',
      label: '$c = \\text{T}$',
      tone: 0,
      dashed: true,
      around: ['eq', 'dT'],
      pad: 0.35,
      labelAt: 'bottom-right',
    },
    {
      id: 'gF',
      label: '$c = \\text{F}$',
      tone: 1,
      dashed: true,
      around: ['uni', 'dF'],
      pad: 0.35,
      labelAt: 'bottom-left',
    },
    {
      id: 'P',
      label: 'participants $p$',
      tone: 'ink',
      around: ['a', 'sub', 't', 'know', 'c', 'eq', 'uni', 'r'],
      pad: 0.8,
      labelAt: 'bottom-left',
    },
    {
      id: 'Q',
      label: 'questions $q$',
      tone: 'ink',
      around: ['d', 'tau', 'sub', 't', 'know', 'c', 'eq', 'uni', 'r', 'y', 'py'],
      pad: 0.5,
      labelAt: 'bottom-right',
    },
  ],
}

/** The DARE model: ability, difficulty and discrimination decide whether a participant knows the answer; a gate on that decides the response. */
export function DareGraph() {
  return (
    <Interactive
      title="Factor graph of the difficulty–ability–response model"
      caption="Participant p's ability minus question q's difficulty, blurred by noise of precision τ_q, decides whether p knows the answer (c_pq). The two gates, selected by c_pq, say that a participant who knows gives the true answer y_q and one who does not answers uniformly at random. Responses are observed; the true answers need not be. The labelled edge carries the vote that each response casts for the true answer."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph: ability and difficulty give an advantage, a probit gives knowledge, gates on knowledge give the response from the true answer or uniformly"
      />
    </Interactive>
  )
}
