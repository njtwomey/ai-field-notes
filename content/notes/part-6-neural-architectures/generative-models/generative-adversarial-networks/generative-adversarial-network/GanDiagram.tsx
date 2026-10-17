import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'z', x: 0, y: 1, shape: 'noise', label: '$\\zvec$', tone: 'neutral' },
    { id: 'G', x: 2.4, y: 1, w: 2.4, h: 0.9, label: 'generator $G_{\\thetavec}$', tone: 1 },
    { id: 'data', x: 2.4, y: 3.4, w: 2.4, h: 0.9, label: 'data $\\xvec \\sim p_{\\text{data}}$', tone: 'neutral' },
    { id: 'D', x: 7, y: 2.2, w: 2.6, h: 1, label: 'discriminator $D_{\\phivec}$', tone: 0 },
    { id: 'V', x: 12, y: 2.2, w: 2.6, h: 1, label: 'value $V(D, G)$\nbinary cross-entropy', tone: 'neutral' },
  ],
  edges: [
    { from: 'z', to: 'G' },
    { from: 'G:e', to: 'D:n', via: [[7, 1]], label: '$G(\\zvec)$, label 0' },
    {
      from: 'data:e',
      to: 'D:w',
      via: [
        [5, 3.4],
        [5, 2.2],
      ],
      label: '$\\xvec$, label 1',
      labelSide: 'right',
    },
    { from: 'D', to: 'V', label: '$D(\\cdot) \\in (0, 1)$' },
    {
      from: 'V:n',
      to: 'G:n',
      via: [
        [12, -0.3],
        [2.4, -0.3],
      ],
      dashed: true,
      label: 'generator: descend in $\\thetavec$, gradient through $D$',
      labelSide: 'right',
    },
    {
      from: 'V:s',
      to: 'D:s',
      via: [
        [12, 4.3],
        [7, 4.3],
      ],
      dashed: true,
      label: 'discriminator: ascend in $\\phivec$',
    },
  ],
}

/** The two players of a GAN and where each one's gradient comes from. */
export function GanDiagram() {
  return (
    <Figure
      title="The adversarial game"
      caption="The discriminator classifies real data (label 1) against generated samples (label 0) and climbs the value V. The generator never sees the data: its only training signal is the gradient of V passed back through the discriminator into the sample it produced. The two updates alternate."
    >
      <Diagram
        spec={spec}
        ariaLabel="GAN: noise z into generator G, generated samples and real data into discriminator D, D's output into the value V; dashed gradients back to D (ascend) and G (descend)"
      />
    </Figure>
  )
}
