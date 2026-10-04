import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'X', x: 0, y: 1.2, shape: 'latent', label: '$\\Xmat$', tone: 'ink' },
    { id: 'Z', x: 2.6, y: -0.8, shape: 'circle', w: 0.7, h: 0.7, label: '$\\Zmat$', tone: 'ink', small: true },
    { id: 'u', x: 4.2, y: -0.8, shape: 'latent', label: '$\\uvec_d$', tone: 'ink' },
    { id: 'f', x: 4.2, y: 1.2, shape: 'latent', label: '$\\fvec_d$', tone: 'ink' },
    { id: 'y', x: 4.2, y: 3.2, shape: 'circle', filled: true, label: '$\\yvec_{:,d}$', tone: 'ink' },
    { id: 'beta', x: 6.4, y: 3.2, shape: 'circle', w: 0.7, h: 0.7, label: '$\\beta$', tone: 'ink', small: true },
    { id: 'theta', x: 6.4, y: 1.2, shape: 'circle', w: 0.7, h: 0.7, label: '$\\thetavec$', tone: 'ink', small: true },
    {
      id: 'q',
      x: 0,
      y: 3.2,
      w: 3.4,
      shape: 'text',
      small: true,
      label: '$q(\\Xmat) = \\prod_n \\Gauss(\\muvec_n, \\Smat_n)$',
    },
  ],
  edges: [
    { from: 'X', to: 'f', route: 'straight' },
    { from: 'Z', to: 'u', route: 'straight' },
    { from: 'u', to: 'f', route: 'straight' },
    { from: 'f', to: 'y', route: 'straight' },
    { from: 'beta', to: 'y', route: 'straight' },
    { from: 'theta', to: 'f', route: 'straight' },
    { from: 'q', to: 'X', dashed: true, arrow: 'none' },
  ],
  groups: [{ id: 'D', label: '$D$', tone: 'ink', around: ['u', 'f', 'y'], pad: 0.45, labelAt: 'bottom-right' }],
}

/** The GP-LVM augmented with inducing variables, the model on which the variational bound is built. */
export function AugmentedModel() {
  return (
    <Figure
      title="The augmented model behind the bound"
      caption="Each output dimension d has latent function values f_d at the N latent points and M inducing values u_d at inducing inputs Z. The inducing inputs are variational parameters, not random variables. The latent points X are integrated out under a factorised Gaussian q(X); the double circles are the quantities integrated out."
    >
      <Diagram spec={spec} ariaLabel="Graphical model of the Bayesian GP-LVM with inducing variables" />
    </Figure>
  )
}
