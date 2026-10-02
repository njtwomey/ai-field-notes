import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'

/* ----------------------------------------------------------------------------------------------------------------- */
/* The progression of ideas                                                                                           */
/* ----------------------------------------------------------------------------------------------------------------- */

const idea = (id: string, x: number, y: number, label: string, tone: number): DiagramNode => ({
  id,
  x,
  y,
  w: 2.3,
  h: 0.95,
  label,
  tone,
  small: true,
})
const arrow = (from: string, to: string, extra: Partial<DiagramEdge> = {}): DiagramEdge => ({
  from,
  to,
  route: 'straight',
  ...extra,
})

const progression: DiagramSpec = {
  spread: [1.4, 1],
  unit: 34,
  nodes: [
    { id: 'row0', x: -2, y: 0.4, shape: 'text', small: true, label: 'statistical\nphysics' },
    { id: 'row1', x: -2, y: 2.2, shape: 'text', small: true, label: 'associative\nmemory' },
    { id: 'row2', x: -2, y: 3.9, shape: 'text', small: true, label: 'learning a\ndistribution' },
    idea('ising', 0, 0.4, 'Ising model\n1925', 0),
    idea('mc', 2.6, 1.3, 'Metropolis 1953\nGlauber 1963', 0),
    idea('sk', 5.2, 0.4, 'spin glasses\n1975', 0),
    idea('sa', 7.8, 0.4, 'simulated\nannealing 1983', 0),
    idea('hebb', 0, 2.2, 'Hebb’s rule\n1949', 1),
    idea('hopfield', 5.2, 2.2, 'Hopfield network\n1982', 1),
    idea('capacity', 10.4, 2.2, 'capacity $0.138N$\n1985', 1),
    idea('dam', 13, 2.2, 'dense associative\nmemory 2016', 1),
    idea('mhn', 15.6, 2.2, 'Hopfield layers\n= attention 2020', 1),
    idea('bm', 7.8, 3.9, 'Boltzmann\nmachine 1985', 2),
    idea('rbm', 10.4, 3.9, 'harmonium\n(RBM) 1986', 2),
    idea('cd', 13, 3.9, 'contrastive\ndivergence 2002', 2),
    idea('dbn', 15.6, 3.9, 'deep belief\nnets 2006', 2),
  ],
  edges: [
    arrow('ising', 'sk'),
    arrow('ising', 'mc'),
    { from: 'mc', to: 'sa', via: [[7.8, 1.3]] },
    arrow('sk', 'hopfield'),
    arrow('hebb', 'hopfield'),
    arrow('hopfield', 'capacity'),
    arrow('sk', 'capacity'),
    arrow('capacity', 'dam'),
    arrow('dam', 'mhn'),
    arrow('sa', 'bm'),
    arrow('hopfield', 'bm'),
    arrow('bm', 'rbm'),
    arrow('rbm', 'cd'),
    arrow('cd', 'dbn'),
  ],
}

export function ProgressionDiagram() {
  return (
    <Interactive
      title="From spins to Boltzmann machines and attention"
      caption="Top row: the statistical physics of interacting spins. Middle row: networks that store memories as energy minima. Bottom row: networks that learn a probability distribution. An arrow means that the later idea uses the earlier one. The 2024 Nobel Prize in Physics cited the Hopfield network and the Boltzmann machine."
    >
      <Diagram
        spec={progression}
        ariaLabel="Timeline: Ising model 1925 leads to Monte Carlo dynamics and spin glasses; spin glasses and Hebb's rule lead to the Hopfield network 1982; Hopfield network and simulated annealing lead to the Boltzmann machine 1985, then the RBM 1986, contrastive divergence 2002 and deep belief nets 2006; the Hopfield network's capacity analysis 1985 leads to dense associative memory 2016 and Hopfield layers as attention 2020"
      />
    </Interactive>
  )
}

/* ----------------------------------------------------------------------------------------------------------------- */
/* Frustration                                                                                                        */
/* ----------------------------------------------------------------------------------------------------------------- */

const triangle: DiagramSpec = {
  unit: 44,
  nodes: [
    variable('a', 0, 2, '$+1$', { filled: true }),
    variable('b', 3, 2, '$-1$', { filled: true }),
    variable('c', 1.5, 0, '$?$'),
    { id: 'la', x: 0, y: 2.9, shape: 'text', small: true, label: '$s_1$' },
    { id: 'lb', x: 3, y: 2.9, shape: 'text', small: true, label: '$s_2$' },
    { id: 'lc', x: 1.5, y: -0.9, shape: 'text', small: true, label: '$s_3$' },
  ],
  edges: [
    link('a', 'b', false, { label: '$J = -1$ satisfied', labelSide: 'right' }),
    link('a', 'c', false, { label: '$J = -1$' }),
    link('b', 'c', false, { label: '$J = -1$', labelSide: 'right' }),
  ],
}

export function FrustratedTriangle() {
  return (
    <Interactive
      title="A frustrated triangle"
      caption="Three spins joined by antiferromagnetic couplings J = −1, each of which prefers its two spins to differ. Once s₁ and s₂ differ, s₃ equals one of them whichever sign it takes, so one bond is always unsatisfied. Six of the eight states tie for the lowest energy, −1."
    >
      <Diagram
        spec={triangle}
        ariaLabel="Triangle of three spins with antiferromagnetic couplings; the third spin cannot satisfy both of its bonds"
      />
    </Interactive>
  )
}

/* ----------------------------------------------------------------------------------------------------------------- */
/* Architectures                                                                                                      */
/* ----------------------------------------------------------------------------------------------------------------- */

function layer(prefix: string, count: number, x0: number, y: number, label: string, observed: boolean): DiagramNode[] {
  return Array.from({ length: count }, (_, k) =>
    variable(`${prefix}${k}`, x0 + 1.4 * k, y, `$${label}_{${k + 1}}$`, { filled: observed }),
  )
}

const pairs = (a: string[], b: string[]) => a.flatMap((x) => b.map((y): [string, string] => [x, y]))

const general = (() => {
  const v = layer('gv', 3, 0.7, 3, 'v', true)
  const h = layer('gh', 2, 1.4, 0.6, 'h', false)
  const ids = [...v, ...h].map((n) => n.id)
  const edges = ids.flatMap((a, i) => ids.slice(i + 1).map((b) => link(a, b, false)))
  return { nodes: [...v, ...h], edges }
})()

const restricted = (() => {
  const v = layer('rv', 4, 5.6, 3, 'v', true)
  const h = layer('rh', 3, 6.3, 0.6, 'h', false)
  const edges = pairs(
    v.map((n) => n.id),
    h.map((n) => n.id),
  ).map(([a, b]) => link(a, b, false))
  return { nodes: [...v, ...h], edges }
})()

const architectures: DiagramSpec = {
  unit: 40,
  nodes: [...general.nodes, ...restricted.nodes],
  edges: [...general.edges, ...restricted.edges],
  groups: [
    {
      id: 'g1',
      rect: { x: -0.2, y: -0.4, w: 4.6, h: 4.4 },
      label: 'Boltzmann machine',
      labelAt: 'bottom-right',
      dashed: true,
    },
    {
      id: 'g2',
      rect: { x: 4.7, y: -0.4, w: 6, h: 4.4 },
      label: 'restricted Boltzmann machine',
      labelAt: 'bottom-right',
      dashed: true,
    },
  ],
}

export function BoltzmannArchitectures() {
  return (
    <Interactive
      title="General and restricted Boltzmann machines"
      caption="Shaded circles are visible units, which the data clamp; open circles are hidden units. Every line is a symmetric weight. A general Boltzmann machine may connect any pair of units. A restricted Boltzmann machine connects only visible to hidden, so the units of one layer are conditionally independent given the other."
    >
      <Diagram
        spec={architectures}
        ariaLabel="Left: three visible and two hidden units with every pair connected. Right: four visible and three hidden units with connections only between the layers"
      />
    </Interactive>
  )
}

const dbn = (() => {
  const v = layer('v', 4, 0, 6, 'v', true)
  const h1 = layer('a', 4, 0, 4, 'h^{(1)}', false)
  const h2 = layer('b', 3, 0.7, 2, 'h^{(2)}', false)
  const h3 = layer('c', 3, 0.7, 0, 'h^{(3)}', false)
  const ids = (ns: DiagramNode[]) => ns.map((n) => n.id)
  const edges = [
    ...pairs(ids(h3), ids(h2)).map(([a, b]) => link(a, b, false)),
    ...pairs(ids(h2), ids(h1)).map(([a, b]) => link(a, b, true)),
    ...pairs(ids(h1), ids(v)).map(([a, b]) => link(a, b, true)),
  ]
  const labels: DiagramNode[] = [
    { id: 't3', x: 6.4, y: 1, shape: 'text', small: true, label: 'RBM 3: undirected\ntop-level associative memory' },
    {
      id: 't2',
      x: 6.4,
      y: 3,
      shape: 'text',
      small: true,
      label: 'RBM 2, trained on samples of $\\hvec^{(1)}$;\nkept as directed weights',
    },
    {
      id: 't1',
      x: 6.4,
      y: 5,
      shape: 'text',
      small: true,
      label: 'RBM 1, trained on the data;\nkept as directed weights',
    },
  ]
  return { nodes: [...v, ...h1, ...h2, ...h3, ...labels], edges }
})()

const dbnSpec: DiagramSpec = { unit: 40, nodes: dbn.nodes, edges: dbn.edges }

export function DeepBeliefNetDiagram() {
  return (
    <Interactive
      title="A deep belief net built from three RBMs"
      caption="Each RBM is trained on the hidden activities of the one below it. In the finished model the top two layers remain an undirected RBM; every lower layer of weights becomes a directed generative connection pointing down, towards the data. Recognition runs the same weights upwards."
    >
      <Diagram
        spec={dbnSpec}
        ariaLabel="Four layers: visible units at the bottom, then three hidden layers. The top two hidden layers are joined by undirected edges; lower layers by arrows pointing down"
      />
    </Interactive>
  )
}
