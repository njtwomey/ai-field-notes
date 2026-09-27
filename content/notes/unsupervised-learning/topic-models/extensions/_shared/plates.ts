/**
 * Plate diagrams for the extensions of LDA, drawn in the style of the shared `lda` spec: hyperparameters are small
 * circles, observed variables are shaded, and plates are labelled bottom-right with their size.
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

/** The dynamic topic model: three time slices, with topics and topic-proportion means chained by Gaussian drift. */
export const dtm: DiagramSpec = (() => {
  const slices = [
    { s: 'a', t: 't-1', x: 0 },
    { s: 'b', t: 't', x: 3.4 },
    { s: 'c', t: 't+1', x: 6.8 },
  ]
  return {
    nodes: slices.flatMap(({ s, t, x }) => [
      variable(`alpha${s}`, x, 0, `$\\alphavec_{${t}}$`),
      variable(`eta${s}`, x, 1.7, '$\\etavec_d$'),
      variable(`z${s}`, x, 3.3, '$z_{dn}$'),
      variable(`w${s}`, x, 4.9, '$w_{dn}$', { filled: true }),
      variable(`beta${s}`, x, 7, `$\\betavec_{${t},k}$`),
    ]),
    edges: [
      ...slices.flatMap(({ s }) => [
        link(`alpha${s}`, `eta${s}`),
        link(`eta${s}`, `z${s}`),
        link(`z${s}`, `w${s}`),
        link(`beta${s}`, `w${s}`),
      ]),
      link('alphaa', 'alphab'),
      link('alphab', 'alphac'),
      link('betaa', 'betab'),
      link('betab', 'betac'),
    ],
    groups: [
      ...slices.flatMap(({ s }) => [
        plate(`N${s}`, '$N_d$', [`z${s}`, `w${s}`]),
        plate(`D${s}`, '$D_t$', [`eta${s}`, `z${s}`, `w${s}`], 0.7),
      ]),
      { id: 'K', label: '$K$', tone: 'ink', rect: { x: -1, y: 6.2, w: 8.8, h: 1.6 }, labelAt: 'bottom-right' },
    ],
  }
})()

/** Topics over time: each token carries a timestamp drawn from its topic's beta distribution. */
export const tot: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('theta', 1.7, 2.4, '$\\thetavec_d$'),
    variable('z', 3.5, 2.4, '$z_{dn}$'),
    variable('w', 5.3, 1.5, '$w_{dn}$', { filled: true }),
    variable('t', 5.3, 3.3, '$t_{dn}$', { filled: true }),
    variable('phi', 8.2, 1.5, '$\\phivec_k$'),
    variable('psi', 8.2, 3.3, '$\\psivec_k$'),
    hyper('beta', 10, 1.5, '$\\beta$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'w'),
    link('z', 't'),
    link('phi', 'w'),
    link('psi', 't'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['z', 'w', 't']),
    plate('D', '$D$', ['theta', 'z', 'w', 't'], 0.7),
    plate('K', '$K$', ['phi', 'psi']),
  ],
}

/** The author-topic model: each word picks one of the document's authors, then a topic from that author's mixture. */
export const authorTopic: DiagramSpec = {
  nodes: [
    variable('a', 0, 2.4, '$\\avec_d$', { filled: true }),
    variable('x', 1.8, 2.4, '$x_{dn}$'),
    variable('z', 3.6, 2.4, '$z_{dn}$'),
    variable('w', 5.4, 2.4, '$w_{dn}$', { filled: true }),
    variable('theta', 3.6, -0.8, '$\\thetavec_a$'),
    hyper('alpha', 1.6, -0.8, '$\\alpha$'),
    variable('phi', 5.4, 5.5, '$\\phivec_k$'),
    hyper('beta', 7.4, 5.5, '$\\beta$'),
  ],
  edges: [
    link('a', 'x'),
    link('x', 'z'),
    link('theta', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('alpha', 'theta'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['x', 'z', 'w']),
    plate('D', '$D$', ['a', 'x', 'z', 'w'], 0.7),
    plate('A', '$A$', ['theta']),
    plate('K', '$K$', ['phi']),
  ],
}

/** The correlated topic model: LDA with a logistic-normal prior on the topic proportions. */
export const ctm: DiagramSpec = {
  nodes: [
    hyper('mu', 0, 1.7, '$\\muvec$'),
    hyper('Sigma', 0, 3.1, '$\\Sigmamat$'),
    variable('eta', 1.8, 2.4, '$\\etavec_d$'),
    variable('z', 3.6, 2.4, '$z_{dn}$'),
    variable('w', 5.4, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.4, -0.8, '$\\phivec_k$'),
    hyper('beta', 7.4, -0.8, '$\\beta$'),
  ],
  edges: [
    link('mu', 'eta'),
    link('Sigma', 'eta'),
    link('eta', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('beta', 'phi'),
  ],
  groups: [plate('N', '$N_d$', ['z', 'w']), plate('D', '$D$', ['eta', 'z', 'w'], 0.75), plate('K', '$K$', ['phi'])],
}

/** Supervised LDA: a response per document, regressed on the document's empirical topic frequencies. */
export const slda: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 1.6, '$\\alpha$'),
    variable('theta', 1.7, 1.6, '$\\thetavec_d$'),
    variable('z', 3.5, 1.6, '$z_{dn}$'),
    variable('w', 5.3, 1.6, '$w_{dn}$', { filled: true }),
    variable('y', 3.5, 3.9, '$y_d$', { filled: true }),
    variable('coef', 7.4, 3.3, '$\\etavec$'),
    hyper('sigma', 7.4, 4.5, '$\\sigma^2$'),
    variable('phi', 5.3, -1.5, '$\\phivec_k$'),
    hyper('beta', 7.3, -1.5, '$\\beta$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'w'),
    link('z', 'y'),
    link('coef', 'y'),
    link('sigma', 'y'),
    link('phi', 'w'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['z', 'w']),
    plate('D', '$D$', ['theta', 'z', 'w', 'y'], 0.7),
    plate('K', '$K$', ['phi']),
  ],
}

/** Labelled LDA: the observed label set of a document restricts which topics its words may use. */
export const labelledLda: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('lambda', 1.7, 0.6, '$\\Lambda_d$', { filled: true }),
    variable('theta', 1.7, 2.4, '$\\thetavec_d$'),
    variable('z', 3.5, 2.4, '$z_{dn}$'),
    variable('w', 5.3, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.3, 5.4, '$\\phivec_k$'),
    hyper('beta', 7.3, 5.4, '$\\beta$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('lambda', 'theta'),
    link('theta', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['z', 'w']),
    plate('D', '$D$', ['lambda', 'theta', 'z', 'w'], 0.7),
    plate('K', '$K$', ['phi']),
  ],
}

/** HDP topic model in its stick-breaking form: global topic weights, per-document weights, infinitely many topics. */
export const hdp: DiagramSpec = {
  nodes: [
    hyper('gamma', -1.8, 2.4, '$\\gamma$'),
    variable('global', 0, 2.4, '$\\betavec$'),
    hyper('alpha', 1.7, 0.4, '$\\alpha_0$'),
    variable('pi', 1.9, 2.4, '$\\pivec_d$'),
    variable('z', 3.7, 2.4, '$z_{dn}$'),
    variable('w', 5.5, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.5, -0.9, '$\\phivec_k$'),
    hyper('eta', 7.5, -0.9, '$\\eta$'),
  ],
  edges: [
    link('gamma', 'global'),
    link('global', 'pi'),
    link('alpha', 'pi'),
    link('pi', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('eta', 'phi'),
  ],
  groups: [plate('N', '$N_d$', ['z', 'w']), plate('D', '$D$', ['pi', 'z', 'w'], 0.7), plate('K', '$\\infty$', ['phi'])],
}

/** Hierarchical LDA: each document draws a root-to-leaf path from the nested CRP, and each word a level on it. */
export const hlda: DiagramSpec = {
  nodes: [
    hyper('gamma', 0, 0.6, '$\\gamma$'),
    variable('c', 1.7, 0.6, '$\\cvec_d$'),
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('theta', 1.7, 2.4, '$\\thetavec_d$'),
    variable('z', 3.5, 2.4, '$z_{dn}$'),
    variable('w', 5.3, 2.4, '$w_{dn}$', { filled: true }),
    variable('phi', 5.3, -1.9, '$\\phivec_k$'),
    hyper('eta', 7.3, -1.9, '$\\eta$'),
  ],
  edges: [
    link('gamma', 'c'),
    link('c', 'w'),
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('eta', 'phi'),
  ],
  groups: [
    plate('N', '$N_d$', ['z', 'w']),
    plate('D', '$D$', ['c', 'theta', 'z', 'w'], 0.7),
    plate('K', '$\\infty$', ['phi']),
  ],
}

/** The structural topic model: covariates shift topic prevalence (through Γ) and topic content (through κ). */
export const stm: DiagramSpec = {
  nodes: [
    hyper('Gamma', 1.8, -0.9, '$\\Gammamat$'),
    hyper('Sigma', 1.8, 4.7, '$\\Sigmamat$'),
    variable('x', 0, 2.4, '$\\xvec_d$', { filled: true }),
    variable('eta', 1.8, 2.4, '$\\etavec_d$'),
    variable('z', 3.6, 2.4, '$z_{dn}$'),
    variable('w', 5.4, 2.4, '$w_{dn}$', { filled: true }),
    variable('y', 5.4, 0.6, '$y_d$', { filled: true }),
    variable('kappa', 7.8, 2.4, '$\\kappavec$'),
  ],
  edges: [
    link('Gamma', 'eta'),
    link('Sigma', 'eta'),
    link('x', 'eta'),
    link('eta', 'z'),
    link('z', 'w'),
    link('y', 'w'),
    link('kappa', 'w'),
  ],
  groups: [plate('N', '$N_d$', ['z', 'w']), plate('D', '$D$', ['x', 'eta', 'z', 'w', 'y'], 0.7)],
}

/** The biterm topic model: one corpus-level topic mixture; each biterm draws one topic and then both its words. */
export const btm: DiagramSpec = {
  nodes: [
    hyper('alpha', 0, 2.4, '$\\alpha$'),
    variable('theta', 1.7, 2.4, '$\\thetavec$'),
    variable('z', 3.6, 2.4, '$z_b$'),
    variable('wi', 5.4, 1.5, '$w_{b1}$', { filled: true }),
    variable('wj', 5.4, 3.3, '$w_{b2}$', { filled: true }),
    variable('phi', 8.2, 2.4, '$\\phivec_k$'),
    hyper('beta', 10, 2.4, '$\\beta$'),
  ],
  edges: [
    link('alpha', 'theta'),
    link('theta', 'z'),
    link('z', 'wi'),
    link('z', 'wj'),
    link('phi', 'wi'),
    link('phi', 'wj'),
    link('beta', 'phi'),
  ],
  groups: [plate('B', '$\\abs{\\Bcal}$', ['z', 'wi', 'wj'], 0.4), plate('K', '$K$', ['phi'])],
}

/** Four-level pachinko allocation: a word walks root → super-topic → sub-topic → word. */
export const pam: DiagramSpec = {
  nodes: [
    hyper('alphar', 0, 1.2, '$\\alphavec_0$'),
    variable('thetar', 1.7, 1.2, '$\\thetavec_{d}$'),
    hyper('alphas', 0, 3.9, '$\\alphavec_s$'),
    variable('thetas', 1.7, 3.9, '$\\thetavec_{ds}$'),
    variable('s', 3.8, 1.2, '$s_{dn}$'),
    variable('z', 3.8, 3.9, '$z_{dn}$'),
    variable('w', 5.6, 3.9, '$w_{dn}$', { filled: true }),
    variable('phi', 8.2, 3.9, '$\\phivec_k$'),
    hyper('beta', 10, 3.9, '$\\beta$'),
  ],
  edges: [
    link('alphar', 'thetar'),
    link('thetar', 's'),
    link('s', 'z'),
    link('alphas', 'thetas'),
    link('thetas', 'z'),
    link('z', 'w'),
    link('phi', 'w'),
    link('beta', 'phi'),
  ],
  groups: [
    plate('S', '$S$', ['thetas']),
    plate('N', '$N_d$', ['s', 'z', 'w']),
    plate('D', '$D$', ['thetar', 'thetas', 's', 'z', 'w'], 0.8),
    plate('K', '$K$', ['phi']),
  ],
}
