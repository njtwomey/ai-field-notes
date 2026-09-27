/**
 * Plate diagrams for topic models applied beyond text, drawn in the style of the shared `lda` spec: hyperparameters are
 * small circles, observed variables are shaded, and plates are labelled bottom-right with their size.
 */
import { link, variable } from '@/components/diagram/components'
import type { DiagramGroup, DiagramSpec } from '@/components/diagram/types'

const hyper = (id: string, x: number, y: number, label: string) =>
  variable(id, x, y, label, { w: 0.7, h: 0.7, small: true })

const plate = (id: string, label: string, around: string[], pad = 0.3): DiagramGroup => ({
  id,
  label,
  tone: 'ink',
  around,
  pad,
  labelAt: 'bottom-right',
})

/**
 * ADL™: one activity-topic per sensor document, generating a bag of unigrams and a first-order chain of bigrams, each
 * from its own topic-specific distribution.
 */
export const adltm: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('theta', 1.7, 2.4, '$\\thetavec$'),
    variable('z', 3.6, 2.4, '$z_d$'),
    variable('t', 5.8, 1, '$t_{dm}$', { filled: true }),
    variable('wprev', 5.8, 3.8, '$w_{d,n-1}$', { filled: true }),
    variable('w', 7.8, 3.8, '$w_{dn}$', { filled: true }),
    variable('nu', 10.4, 1, '$\\nuvec_k$'),
    hyper('gamma', 12.2, 1, '$\\gamma$'),
    variable('phi', 10.4, 3.8, '$\\Phimat_k$'),
    hyper('beta', 12.2, 3.8, '$\\beta$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 't'),
    link('z', 'w'),
    link('wprev', 'w'),
    link('nu', 't'),
    link('phi', 'w'),
    link('gamma', 'nu'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('M', '$M_d$', ['t']),
    plate('N', '$N_d$', ['wprev', 'w']),
    plate('D', '$D$', ['z', 't', 'wprev', 'w'], 0.7),
    plate('K1', '$K$', ['nu']),
    plate('K2', '$K$', ['phi']),
  ],
}

/** STRUCTURE with admixture: individuals are documents, allele copies are words, populations are topics. */
export const structure: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('q', 1.7, 2.4, '$\\qvec_i$'),
    variable('z', 3.6, 2.4, '$z_{\\ell a}^{(i)}$'),
    variable('x', 5.6, 2.4, '$x_{\\ell a}^{(i)}$', { filled: true }),
    variable('p', 5.6, -1.2, '$\\pvec_{k\\ell}$'),
    hyper('lambda', 7.8, -1.2, '$\\lambda$'),
  ],
  edges: [link('alpha', 'q'), link('q', 'z'), link('z', 'x'), link('p', 'x'), link('lambda', 'p')],
  groups: [
    plate('A', '$2$', ['z', 'x'], 0.3),
    plate('L', '$L$', ['z', 'x'], 0.6),
    plate('N', '$N$', ['q', 'z', 'x'], 0.9),
    plate('Kp', '$K$', ['p'], 0.3),
    plate('Lp', '$L$', ['p'], 0.6),
  ],
}

/** Collaborative topic regression: an item's latent vector is its topic proportions plus an offset learned from users. */
export const ctr: DiagramSpec = {
  nodes: [
    hyper('alpha', -0.4, 1.2, '$\\alpha$'),
    variable('theta', 1.7, 1.2, '$\\thetavec_j$'),
    variable('z', 3.6, 1.2, '$z_{jn}$'),
    variable('w', 5.5, 1.2, '$w_{jn}$', { filled: true }),
    variable('phi', 5.5, -1.8, '$\\phivec_k$'),
    variable('v', 1.7, 3.4, '$\\vvec_j$'),
    hyper('lv', -0.4, 3.4, '$\\lambda_v$'),
    variable('r', 1.7, 5.7, '$r_{ij}$', { filled: true }),
    variable('u', -1.6, 5.7, '$\\uvec_i$'),
    hyper('lu', -3.4, 5.7, '$\\lambda_u$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('theta', 'v'),
    link('lv', 'v'),
    link('v', 'r'),
    link('u', 'r'),
    link('lu', 'u'),
  ],
  groups: [
    plate('N', '$N_j$', ['z', 'w']),
    plate('J', '$J$', ['theta', 'z', 'w', 'v', 'r'], 0.75),
    plate('I', '$I$', ['u', 'r'], 0.4),
    plate('K', '$K$', ['phi']),
  ],
}

/** Fei-Fei and Perona's scene model: an observed category selects the prior of each image's mixture of themes. */
export const sceneModel: DiagramSpec = {
  nodes: [
    variable('c', 0, 2.4, '$c_d$', { filled: true }),
    variable('alpha', 0, -0.9, '$\\alphavec_c$', { w: 0.9, h: 0.9 }),
    variable('theta', 1.9, 2.4, '$\\thetavec_d$'),
    variable('z', 3.8, 2.4, '$z_{dn}$'),
    variable('x', 5.7, 2.4, '$x_{dn}$', { filled: true }),
    variable('phi', 5.7, -0.9, '$\\phivec_k$'),
    hyper('beta', 7.6, -0.9, '$\\beta$'),
  ],
  edges: [
    link('c', 'theta'),
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'x'),
    link('phi', 'x'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['z', 'x']),
    plate('D', '$D$', ['c', 'theta', 'z', 'x'], 0.7),
    plate('C', '$C$', ['alpha']),
    plate('K', '$K$', ['phi']),
  ],
}
