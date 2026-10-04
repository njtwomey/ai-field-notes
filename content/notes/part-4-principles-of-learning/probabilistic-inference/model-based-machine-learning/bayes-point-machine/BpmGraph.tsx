import { Diagram, factor, Figure, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 44,
  nodes: [
    factor('prior', 0, 0, '$\\Gauss(\\zeros, \\Imat)$', 'w'),
    variable('w', 0, 1.5, '$\\wvec$'),
    factor('ip', 2.5, 1.5, '$\\times$', 'n'),
    variable('x', 2.5, 3, '$\\xvec_n$', { filled: true }),
    variable('a', 4.2, 1.5, '$a_n$'),
    factor('lik', 6, 1.5, '$\\Phi(y_n a_n / \\beta)$', 'n'),
    variable('y', 7.7, 1.5, '$y_n$', { filled: true }),
  ],
  edges: [
    line('prior', 'w'),
    line('w', 'ip', '$\\tilde t_n(\\wvec)$'),
    line('x', 'ip'),
    line('ip', 'a'),
    line('a', 'lik', 'site in $a_n$'),
    line('lik', 'y'),
  ],
  groups: [
    {
      id: 'N',
      label: 'points $n$',
      tone: 'ink',
      around: ['ip', 'x', 'a', 'lik', 'y'],
      pad: 0.5,
      labelAt: 'bottom-right',
    },
  ],
}

/** The Bayes point machine as a factor graph, with the EP site of each point on its edge into the weights. */
export function BpmGraph() {
  return (
    <Figure
      title="Factor graph of the Bayes point machine"
      caption="One weight vector w with a N(0, I) prior is shared by every point. Point n forms the activation aₙ = wᵀxₙ and its label yₙ has probability Φ(yₙaₙ/β). Inputs and labels are observed. EP replaces each probit factor by a Gaussian site in aₙ, which reaches w as the message t̃ₙ(w)."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph: weight prior, inner product with each observed input, probit factor to each observed label"
      />
    </Figure>
  )
}
