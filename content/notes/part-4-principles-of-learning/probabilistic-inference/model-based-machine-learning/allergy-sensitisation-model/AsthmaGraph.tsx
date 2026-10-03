import { factor, link, variable } from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec } from 'aifn-render'
import { StepGraph, type GraphStep } from '../_shared/StepGraph'

const AGES = [1, 3, 5, 8]
const X = (i: number) => 1 + i * 2.6
const line = (a: string, b: string) => link(a, b, false)

const nodes: DiagramNode[] = [
  variable('cls', -1.2, -0.6, '$z_n$'),
  factor('pz', -1.2, -2.4, '$\\Cat$', 'w'),
  { id: 'gdot', x: 0.2, y: -0.6, shape: 'dot', w: 0.18, h: 0.18 },
  variable('chain', 4.9, -2.2, '$\\thetavec_{a z}$', { w: 1.1, h: 1.1 }),
  variable('tskin', 2.3, 6.6, '$\\phivec_{\\text{skin}}$', { w: 1.1, h: 1.1 }),
  variable('tige', 7.5, 6.6, '$\\phivec_{\\text{IgE}}$', { w: 1.1, h: 1.1 }),
]
const edges: DiagramEdge[] = [line('pz', 'cls'), line('cls', 'gdot')]
AGES.forEach((a, i) => {
  nodes.push(variable(`s${i}`, X(i), 2, `$s_{${a}}$`))
  nodes.push(factor(`fs${i}`, X(i) - 0.55, 3.3, ''))
  nodes.push(factor(`fi${i}`, X(i) + 0.55, 3.3, ''))
  nodes.push(variable(`k${i}`, X(i) - 0.55, 4.6, '', { filled: true, w: 0.6, h: 0.6 }))
  nodes.push(variable(`g${i}`, X(i) + 0.55, 4.6, '', { filled: true, w: 0.6, h: 0.6 }))
  nodes.push(factor(`c${i}`, i === 0 ? X(0) : X(i) - 1.3, i === 0 ? 0.8 : 2, '', 'n'))
  edges.push(line(`s${i}`, `fs${i}`), line(`s${i}`, `fi${i}`), line(`fs${i}`, `k${i}`), line(`fi${i}`, `g${i}`))
  edges.push(line('tskin', `fs${i}`), line('tige', `fi${i}`))
  edges.push(line(`c${i}`, `s${i}`), line('chain', `c${i}`))
  if (i > 0) edges.push(line(`s${i - 1}`, `c${i}`))
})

const SPEC: DiagramSpec = {
  unit: 38,
  nodes,
  edges,
  groups: [
    {
      id: 'gate',
      label: 'class $z_n = k$',
      tone: 1,
      dashed: true,
      around: ['c0', 'c1', 'c2', 'c3', 'gdot'],
      pad: 0.45,
      labelAt: 'top-left',
    },
    {
      id: 'A',
      label: 'allergens $a$',
      tone: 'ink',
      around: ['chain', 's0', 's3', 'k0', 'g3', 'c0'],
      pad: 0.6,
      labelAt: 'bottom-right',
    },
    {
      id: 'N',
      label: 'children $n$',
      tone: 'ink',
      around: ['cls', 's0', 's3', 'k0', 'g3', 'gdot'],
      pad: 0.9,
      labelAt: 'bottom-left',
    },
  ],
}

const tests = AGES.flatMap((_, i) => [`s${i}`, `fs${i}`, `fi${i}`, `k${i}`, `g${i}`])
const chain = AGES.map((_, i) => `c${i}`)

const STEPS: GraphStep[] = [
  {
    add: [...tests, 'A', 'N'],
    text: 'Assumptions 1 and 2: at each age a child is sensitised to an allergen or not, $s_t$, and each skin-prick test and IgE test (small shaded nodes) is positive with high probability if sensitised and low probability if not.',
  },
  {
    add: ['tskin', 'tige'],
    text: 'Assumption 3: the false-positive and false-negative rates of each kind of test are the same across the study, so one parameter node per test type feeds every test factor.',
  },
  {
    add: [...chain],
    text: 'Assumption 4: sensitisation at one age depends only on sensitisation at the previous age: a Markov chain, one factor per step.',
  },
  {
    add: ['chain'],
    text: 'Assumption 5: the probabilities of initially having, gaining and retaining sensitisation to an allergen, $\\thetavec_a$, are the same for every child.',
  },
  {
    add: ['cls', 'pz', 'gdot', 'gate'],
    text: "Sensitisation classes: a latent class $z_n$ per child selects which set of chain probabilities applies. The gate switches the chain factors to class $z_n$'s parameters.",
  },
]

/** The sensitisation model, one assumption at a time, ending with the class gate. */
export function AsthmaGraph() {
  return (
    <StepGraph
      title="Building the sensitisation model"
      caption="Step through the assumptions with the arrows. For one child and one allergen, four ages each have a sensitisation state and two tests; test results are observed and shaded. The last step adds the class gate that turns the model into a mixture of sensitisation patterns."
      spec={SPEC}
      steps={STEPS}
      ariaLabel="Factor graph: a Markov chain of sensitisation states over four ages, each with two observed tests, shared test parameters, and a class gate on the chain parameters"
    />
  )
}
