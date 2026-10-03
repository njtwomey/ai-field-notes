import { Diagram } from 'aifn-render'
import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'right', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 38,
  nodes: [
    variable('u', 0, 0, '$u_{ki}$'),
    variable('x', 0, 1.5, '$x_i$', { filled: true }),
    factor('su', 1.6, 0.75, '$\\Sigma$', 'n'),
    variable('s', 3.1, 0.75, '$s_k$'),
    variable('v', 0, 3.3, '$v_{kj}$'),
    variable('y', 0, 4.8, '$y_j$', { filled: true }),
    factor('sv', 1.6, 4.05, '$\\Sigma$', 's'),
    variable('t', 3.1, 4.05, '$t_k$'),
    factor('mul', 4.4, 2.4, '$\\times$', 'n'),
    variable('z', 5.7, 2.4, '$z_k$'),
    factor('sum', 7.3, 2.4, '$\\Sigma_k$', 'n'),
    variable('rt', 8.7, 2.4, '$\\tilde r$'),
    factor('noise', 10, 2.4, '$\\Gauss(\\tilde r, \\beta^2)$', 's'),
    variable('r', 11.3, 2.4, '$r$'),
    variable('b', 12.6, 0.6, '$\\bvec_u$'),
    factor('ord', 12.6, 2.4, 'thresholds', 's'),
    variable('l', 13.9, 2.4, '$\\ell$', { filled: true }),
  ],
  edges: [
    line('u', 'su'),
    line('x', 'su'),
    line('su', 's'),
    line('v', 'sv'),
    line('y', 'sv'),
    line('sv', 't'),
    line('s', 'mul', '$\\mu_{\\times \\to s_k}$ (VMP)'),
    line('t', 'mul'),
    line('mul', 'z', '$\\mu_{\\times \\to z_k}$'),
    line('z', 'sum'),
    line('sum', 'rt'),
    line('rt', 'noise'),
    line('noise', 'r'),
    line('r', 'ord'),
    line('b', 'ord'),
    line('ord', 'l'),
  ],
  groups: [
    { id: 'I', label: '$i$', tone: 'ink', around: ['u', 'x'], pad: 0.25, labelAt: 'bottom-right' },
    { id: 'J', label: '$j$', tone: 'ink', around: ['v', 'y'], pad: 0.25, labelAt: 'bottom-right' },
    {
      id: 'K',
      label: 'traits $k$',
      tone: 'ink',
      around: ['u', 'x', 'su', 's', 'v', 'y', 'sv', 't', 'mul', 'z'],
      pad: 0.7,
      labelAt: 'bottom-right',
    },
  ],
}

/** Matchbox's bilinear model as a factor graph, for one rating of one item by one user. */
export function MatchboxGraph() {
  return (
    <Interactive
      title="Factor graph of Matchbox for one rating"
      caption="User features xᵢ and item features yⱼ map through weight matrices U and V to trait vectors s and t. Each trait pair is multiplied, the products are summed into an affinity r̃, Gaussian noise gives the latent rating r, and the user's own thresholds bᵤ turn r into the observed star rating ℓ. Every factor passes Gaussian messages exactly except the product factor and the threshold factors."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph of Matchbox: feature weights sum to user and item traits, which multiply and sum to a noisy rating compared with user thresholds"
      />
    </Interactive>
  )
}
