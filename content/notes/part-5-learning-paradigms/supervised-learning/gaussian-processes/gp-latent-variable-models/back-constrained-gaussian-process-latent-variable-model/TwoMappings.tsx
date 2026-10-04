import { Diagram, Figure, projector } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'y', x: 0, y: 2, w: 0.9, h: 0.9, label: '$\\yvec_n$' },
    projector('encoder', 'g', 2.3, 2, 'back constraint\n$\\xvec_n = \\gvec(\\yvec_n; \\Amat)$', { w: 2.4, h: 2.4 }),
    { id: 'x', x: 4.8, y: 2, shape: 'latent', label: '$\\xvec_n$', tone: 2 },
    projector('decoder', 'f', 7.3, 2, 'GP mapping\n$f_d \\sim \\GP(0, k)$', { w: 2.4, h: 2.4, tone: 1 }),
    { id: 'yh', x: 9.8, y: 2, w: 0.9, h: 0.9, label: '$\\yvec_n$' },
  ],
  edges: [
    { from: 'y', to: 'g' },
    { from: 'g', to: 'x' },
    { from: 'x', to: 'f' },
    { from: 'f', to: 'yh' },
    {
      from: 'yh:s',
      to: 'y:s',
      via: [
        [9.8, 3.9],
        [0, 3.9],
      ],
      dashed: true,
      arrow: 'none',
      label: 'objective: $\\log p(\\Ymat \\mid \\Xmat = \\gvec(\\Ymat; \\Amat), \\thetavec)$',
    },
  ],
}

/** The back-constrained GP-LVM as an encoder and a probabilistic decoder. */
export function TwoMappings() {
  return (
    <Figure
      title="Two mappings"
      caption="The back constraint is a smooth map from each observation to its latent point, so observations close in the data get latent points close together. The Gaussian process maps latent points back to the data, so latent points close together produce similar observations. The parameters A of the back constraint replace the latent points as the quantities optimised; the objective is still the GP-LVM likelihood."
    >
      <Diagram
        spec={spec}
        ariaLabel="Back-constrained GP-LVM: a back-constraint encoder and a Gaussian process decoder"
      />
    </Figure>
  )
}
