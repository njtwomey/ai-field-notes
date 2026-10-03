import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

const T = ['1', '2', '3', 'T']
const X = [0, 2.4, 4.8, 8.4]

const spec: DiagramSpec = {
  nodes: [
    ...T.map((t, i) => ({
      id: `x${t}`,
      x: X[i],
      y: 1.4,
      shape: 'circle' as const,
      label: `$\\xvec_${t === 'T' ? 'T' : t}$`,
      tone: 'ink' as const,
    })),
    ...T.map((t, i) => ({
      id: `y${t}`,
      x: X[i],
      y: 3.6,
      shape: 'circle' as const,
      filled: true,
      label: `$\\yvec_${t === 'T' ? 'T' : t}$`,
      tone: 'ink' as const,
    })),
    { id: 'dots', x: 6.6, y: 1.4, shape: 'text', label: '$\\cdots$' },
    { id: 'fdyn', x: 3.6, y: -0.6, shape: 'circle', w: 0.8, h: 0.8, label: '$f$', tone: 0, dashed: true },
    { id: 'gobs', x: 10.6, y: 2.5, shape: 'circle', w: 0.8, h: 0.8, label: '$g$', tone: 1, dashed: true },
    { id: 'lf', x: 6.2, y: -0.6, w: 3.6, shape: 'text', small: true, label: 'dynamics $f \\sim \\GP(0, k_X)$' },
    { id: 'lg', x: 10.6, y: 3.5, w: 3.4, shape: 'text', small: true, label: 'observation map\n$g \\sim \\GP(0, k_Y)$' },
  ],
  edges: [
    { from: 'x1', to: 'x2', route: 'straight' },
    { from: 'x2', to: 'x3', route: 'straight' },
    { from: 'x3', to: 'dots', route: 'straight' },
    { from: 'dots', to: 'xT', route: 'straight' },
    ...T.map((t) => ({ from: `x${t}`, to: `y${t}`, route: 'straight' as const })),
    { from: 'fdyn', to: 'x2', route: 'straight', dashed: true, tone: 0 },
    { from: 'fdyn', to: 'x3', route: 'straight', dashed: true, tone: 0 },
    { from: 'gobs', to: 'yT', route: 'straight', dashed: true, tone: 1 },
  ],
}

/** The GPDM: a latent Markov chain with a GP transition function and a GP observation map, both integrated out. */
export function DynamicsChain() {
  return (
    <Interactive
      title="A latent dynamical system with two Gaussian process maps"
      caption="The poses y_t are observed. The latent states x_t form a first-order Markov chain whose transition function f has a Gaussian process prior; the observation map g has another. Both functions (dashed) are integrated out, which couples every latent state to every other through the kernel matrices. The latent trajectory X is optimised."
    >
      <Diagram spec={spec} ariaLabel="Graphical model of the Gaussian process dynamical model" />
    </Interactive>
  )
}
