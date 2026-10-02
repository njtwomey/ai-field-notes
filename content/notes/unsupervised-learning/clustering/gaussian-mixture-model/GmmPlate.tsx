import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 48,
  nodes: [
    { id: 'pi', x: 0, y: 0, shape: 'circle', w: 0.8, h: 0.8, label: '$\\pivec$', tone: 'ink', small: true },
    { id: 'z', x: 2, y: 0, shape: 'circle', label: '$z_i$', tone: 'ink' },
    { id: 'x', x: 4, y: 0, shape: 'circle', filled: true, label: '$\\xvec_i$', tone: 'ink' },
    { id: 'mu', x: 6.6, y: -0.6, shape: 'circle', label: '$\\muvec_j$', tone: 'ink' },
    { id: 'sig', x: 6.6, y: 0.9, shape: 'circle', label: '$\\Sigma_j$', tone: 'ink' },
    // Plate sizes as text nodes: group labels are set in capitals, which would turn n into N.
    { id: 'nl', x: 4.62, y: 0.62, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$n$' },
    { id: 'kl', x: 7.12, y: 1.42, shape: 'text', small: true, w: 0.3, h: 0.3, label: '$k$' },
  ],
  edges: [
    { from: 'pi', to: 'z', route: 'straight' },
    { from: 'z', to: 'x', route: 'straight' },
    { from: 'mu', to: 'x', route: 'straight' },
    { from: 'sig', to: 'x', route: 'straight' },
  ],
  groups: [
    { id: 'n', tone: 'ink', around: ['z', 'x'], pad: 0.4 },
    { id: 'k', tone: 'ink', around: ['mu', 'sig'], pad: 0.3 },
  ],
}

/** The Gaussian mixture model as a plate diagram. */
export function GmmPlate() {
  return (
    <Interactive
      title="The Gaussian mixture as a graphical model"
      caption="Each of the n points draws a component label z_i from the mixing weights π, then draws x_i from that component's Gaussian. Only x_i is observed (shaded). The k component means and covariances sit in their own plate: every point depends on all of them, and z_i selects which one applies."
    >
      <Diagram
        spec={spec}
        ariaLabel="Plate diagram: mixing weights pi point to the latent label z_i, which points to the observed x_i inside a plate over n points; the component means and covariances, in a plate over k components, also point to x_i"
      />
    </Interactive>
  )
}
