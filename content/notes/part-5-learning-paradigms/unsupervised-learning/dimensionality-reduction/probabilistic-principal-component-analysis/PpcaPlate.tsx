import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const param = (id: string, x: number, y: number, label: string) => ({
  id,
  x,
  y,
  shape: 'circle' as const,
  w: 0.8,
  h: 0.8,
  label,
  tone: 'ink' as const,
  small: true,
})

const spec: DiagramSpec = {
  unit: 48,
  nodes: [
    { id: 'z', x: 0, y: 0, shape: 'circle', label: '$\\zvec_i$', tone: 'ink' },
    { id: 'x', x: 2.2, y: 0, shape: 'circle', filled: true, label: '$\\xvec_i$', tone: 'ink' },
    param('W', 4.4, -1.1, '$\\Wmat$'),
    param('mu', 4.6, 0, '$\\muvec$'),
    param('s2', 4.4, 1.1, '$\\sigma^2$'),
    {
      id: 'prior',
      x: 0,
      y: 1.3,
      shape: 'text',
      small: true,
      w: 2.4,
      label: '$\\zvec_i \\sim \\Gauss(\\zeros, \\Imat_q)$',
    },
    // The plate size as a text node: group labels are set in capitals, which would turn n into N.
    { id: 'nl', x: 2.75, y: 1.7, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$n$' },
  ],
  edges: [
    { from: 'z', to: 'x', route: 'straight' },
    { from: 'W', to: 'x', route: 'straight' },
    { from: 'mu', to: 'x', route: 'straight' },
    { from: 's2', to: 'x', route: 'straight' },
  ],
  groups: [{ id: 'n', tone: 'ink', around: ['z', 'x', 'prior'], pad: 0.35 }],
}

/** Probabilistic PCA as a plate diagram. */
export function PpcaPlate() {
  return (
    <Interactive
      title="PPCA as a graphical model"
      caption="Each observation x_i (shaded) is generated from its own q-dimensional latent z_i through the shared loading matrix W, mean μ and isotropic noise variance σ². The plate repeats the pair for every point; the parameters outside it are shared."
    >
      <Diagram
        spec={spec}
        ariaLabel="Plate diagram: the latent z_i points to the observed x_i inside a plate over n points; the shared parameters W, mu and sigma squared also point to x_i"
      />
    </Interactive>
  )
}
