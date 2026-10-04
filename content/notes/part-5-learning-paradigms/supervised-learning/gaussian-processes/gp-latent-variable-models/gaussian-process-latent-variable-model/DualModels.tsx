import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'W', x: 0, y: 0, shape: 'circle', w: 0.8, h: 0.8, label: '$\\Wmat$', tone: 'ink', highlight: true },
    { id: 'x1', x: 2.4, y: 0, shape: 'circle', label: '$\\xvec_n$', tone: 'ink', dashed: true },
    { id: 'y1', x: 2.4, y: 2.2, shape: 'circle', filled: true, label: '$\\yvec_n$', tone: 'ink' },
    {
      id: 't1',
      x: 1.4,
      y: 4.1,
      w: 5,
      shape: 'text',
      small: true,
      label: 'PPCA: integrate out $\\xvec_n$,\noptimise $\\Wmat$',
    },
    { id: 'x2', x: 7.6, y: 0, shape: 'circle', label: '$\\xvec_n$', tone: 'ink', highlight: true },
    { id: 'y2', x: 7.6, y: 2.2, shape: 'circle', filled: true, label: '$\\yvec_n$', tone: 'ink' },
    { id: 'f', x: 10.2, y: 0, shape: 'circle', w: 0.8, h: 0.8, label: '$f$', tone: 'ink', dashed: true },
    {
      id: 'th',
      x: 12,
      y: 0,
      shape: 'circle',
      w: 0.7,
      h: 0.7,
      label: '$\\thetavec$',
      tone: 'ink',
      small: true,
      highlight: true,
    },
    {
      id: 't2',
      x: 9.2,
      y: 4.1,
      w: 6,
      shape: 'text',
      small: true,
      label: 'GP-LVM: integrate out the mapping $f$,\noptimise $\\xvec_n$ and $\\thetavec$',
    },
  ],
  edges: [
    { from: 'W', to: 'y1', route: 'straight' },
    { from: 'x1', to: 'y1', route: 'straight' },
    { from: 'x2', to: 'y2', route: 'straight' },
    { from: 'f', to: 'y2', route: 'straight' },
    { from: 'th', to: 'f', route: 'straight' },
  ],
  groups: [
    { id: 'N1', label: '$N$', tone: 'ink', around: ['x1', 'y1'], pad: 0.4, labelAt: 'bottom-right' },
    { id: 'N2', label: '$N$', tone: 'ink', around: ['x2', 'y2'], pad: 0.4, labelAt: 'bottom-right' },
  ],
}

/** Probabilistic PCA and the GP-LVM as graphical models: which variable is integrated out and which is optimised. */
export function DualModels() {
  return (
    <Figure
      title="Two ways to make PCA probabilistic"
      caption="Shaded nodes are observed. Dashed nodes are integrated out; highlighted nodes are optimised. Probabilistic PCA integrates out the latent point of each observation and optimises the shared linear map. The GP-LVM does the opposite: it integrates out the map, under a Gaussian process prior with hyperparameters θ, and optimises one latent point per observation. With a linear kernel both give the principal subspace."
    >
      <Diagram spec={spec} ariaLabel="Graphical models of probabilistic PCA and of the GP-LVM" />
    </Figure>
  )
}
