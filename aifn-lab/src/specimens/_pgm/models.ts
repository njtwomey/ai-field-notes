/** Models written in aifn's description language for the pgm specimens, with the bindings they are shown with. */
import { dishonestCasino, dist, ldaModel, model, type Bindings, type Model } from 'aifn/pgm'
import { toRows } from 'aifn/tensor'

/** The sprinkler network: conditional probability tables indexed by the parents' values. */
export const sprinkler: Model = model('Sprinkler network', (m) => {
  const cloudy = m.variable('cloudy', dist.Bernoulli(0.5), { label: 'C' })
  const pS = m.constant('pS', [0.5, 0.1], { label: '\\pi_S' })
  const pR = m.constant('pR', [0.2, 0.8], { label: '\\pi_R' })
  const pW = m.constant(
    'pW',
    [
      [0.01, 0.9],
      [0.9, 0.99],
    ],
    { label: '\\pi_W' },
  )
  const s = m.variable('sprinkler', dist.Bernoulli(pS.at(cloudy)), { label: 'S' })
  const r = m.variable('rain', dist.Bernoulli(pR.at(cloudy)), { label: 'R' })
  const pw = m.deterministic('pw', 'index', [pW, s, r], { label: 'p_W' })
  m.observed('wet', dist.Bernoulli(pw), { label: 'W' })
})

/** A two-component Gaussian mixture with unknown means: a plate of components and a plate of points. */
export const mixture: Model = model('Mixture of two Gaussians', (m) => {
  const K = m.size('K')
  const pi = m.constant('π', [0.5, 0.5], { label: '\\boldsymbol{\\pi}' })
  const mu = m.plate('components', K, { label: 'K' }).variable('μ', dist.Normal(0, 10), { label: '\\mu_k' })
  const points = m.plate('points', 'N', { label: 'N' })
  const z = points.variable('z', dist.Categorical(pi), { label: 'z_n' })
  points.observed('x', dist.Normal(mu.at(z), 1), { label: 'x_n' })
})

/** The casino HMM unrolled over four rolls (the language has no Markov plate, so the chain is written out). */
export function casinoModel(n = 4): Model {
  const h = dishonestCasino()
  return model('Dishonest casino, unrolled', (m) => {
    const A = m.constant('A', toRows(h.transition))
    const B = m.constant('B', toRows(h.emission))
    let prev = m.variable('y0', dist.Categorical([0.5, 0.5]), { label: 'y_0' })
    m.observed('x0', dist.Categorical(B.at(prev)), { label: 'x_0' })
    for (let t = 1; t < n; t++) {
      const y = m.variable(`y${t}`, dist.Categorical(A.at(prev)), { label: `y_${t}` })
      m.observed(`x${t}`, dist.Categorical(B.at(y)), { label: `x_${t}` })
      prev = y
    }
  })
}

export type ModelEntry = {
  label: string
  model: Model
  /** Bindings small enough to draw the expanded factor graph. */
  bindings: Bindings
  /** Hand placements for the plate diagram, by node name (grid units). */
  positions?: Record<string, [number, number]>
  /** The variable whose blanket is shown first. */
  focus: string
}

export const MODELS: Record<string, ModelEntry> = {
  lda: {
    label: 'Latent Dirichlet allocation',
    model: ldaModel(),
    bindings: {
      sizes: { K: 2, V: 4 },
      constants: { α: 0.5, β: 0.1 },
      data: {
        w: [
          [0, 1, 1],
          [2, 3],
        ],
      },
    },
    positions: { α: [0, 2.4], θ: [1.7, 2.4], z: [3.5, 2.4], w: [5.3, 2.4], φ: [5.3, -0.4], β: [7.1, -0.4] },
    focus: 'z[0,1]',
  },
  sprinkler: {
    label: 'Sprinkler network',
    model: sprinkler,
    bindings: { data: { wet: 1 } },
    focus: 'sprinkler',
  },
  mixture: {
    label: 'Gaussian mixture',
    model: mixture,
    bindings: { sizes: { K: 2 }, data: { x: [-1.2, 0.4, 2.5] } },
    focus: 'z[1]',
  },
  casino: {
    label: 'Casino HMM (unrolled)',
    model: casinoModel(),
    bindings: { data: { x0: 5, x1: 5, x2: 0, x3: 5 } },
    focus: 'y1',
  },
}
