import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'right', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 44,
  nodes: [
    factor('p0', 1, 0, '$\\Bern(0.5)$', 'w'),
    factor('p1', 5, 0, '$\\Bern(0.5)$', 'e'),
    variable('s0', 1, 1.4, '$s_{\\text{C\\#}}$'),
    variable('s1', 5, 1.4, '$s_{\\text{SQL}}$'),
    factor('n1', 0, 3.8, 'noise', 'w'),
    variable('a1', 0, 5.2, '$a_1$', { filled: true }),
    factor('n2', 6, 3.8, 'noise', 'e'),
    variable('a2', 6, 5.2, '$a_2$', { filled: true }),
    factor('and', 3, 2.6, 'AND', 'e'),
    variable('h3', 3, 3.8, '$h_3$'),
    factor('n3', 3, 5.0, 'noise', 'e'),
    variable('a3', 3, 6.2, '$a_3$', { filled: true }),
  ],
  edges: [
    line('p0', 's0'),
    line('p1', 's1'),
    line('s0', 'n1', '$(0.9, 0.2)$'),
    line('n1', 'a1'),
    line('s1', 'n2'),
    line('n2', 'a2'),
    line('s0', 'and', '$\\mu_{\\text{AND} \\to s_{\\text{C\\#}}}$'),
    line('s1', 'and'),
    line('and', 'h3'),
    line('h3', 'n3', '$\\lambda(h_3) = (0.1, 0.8)$'),
    line('n3', 'a3'),
  ],
}

const STEPS: GraphStep[] = [
  { add: ['s0', 's1'], text: 'Assumption 1: each skill is a binary variable, had or not had.' },
  {
    add: ['p0', 'p1'],
    text: 'Assumption 2: before any answers, each skill has probability 0.5, a prior factor $\\Bern(0.5)$.',
  },
  {
    add: ['n1', 'a1', 'n2', 'a2'],
    text: 'Assumptions 3 and 4: a noise factor per question gives the answer probability, $0.9$ with the skill and $0.2$ without. Questions 1 and 2 test one skill each; answers are observed.',
  },
  {
    add: ['and', 'h3', 'n3', 'a3'],
    text: 'A question needing both skills adds a deterministic AND factor, $h_3 = s_{\\text{C\\#}} s_{\\text{SQL}}$, before its noise factor. Assumption 5, that nothing else affects the answer, is the absence of any other edge.',
  },
  {
    add: [],
    text: 'Messages for answers (right, wrong, wrong). A right answer to Q1 sends $(0.9, 0.2)$ to $s_{\\text{C\\#}}$. A wrong answer to Q3 sends $\\lambda(h_3) = (0.1, 0.8)$ up to the AND factor, which passes $s_{\\text{C\\#}}$ the average of $\\lambda$ over its belief in SQL.',
  },
]

/** The skills factor graph for three questions, one assumption at a time, ending with the messages. */
export function SkillsGraph() {
  return (
    <StepGraph
      title="Building the skills factor graph"
      caption="Step through the assumptions with the arrows. New nodes are highlighted; the answers are observed and shaded. The last step labels the messages derived in the text."
      spec={SPEC}
      steps={STEPS}
      labelsFrom={4}
      ariaLabel="Factor graph of the skills model: two skills with priors, noise factors for single-skill questions, and an AND factor for a two-skill question"
    />
  )
}
