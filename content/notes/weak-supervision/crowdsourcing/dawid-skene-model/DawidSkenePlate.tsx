import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 48,
  nodes: [
    { id: 'p', x: 0, y: 0, shape: 'circle', w: 0.8, h: 0.8, label: '$\\mathbf{p}$', tone: 'ink', small: true },
    { id: 't', x: 2, y: 0, shape: 'circle', label: '$t_i$', tone: 'ink' },
    { id: 'n', x: 4.4, y: 0, shape: 'circle', w: 1, h: 1, filled: true, label: '$n^{(k)}_{i\\cdot}$', tone: 'ink' },
    { id: 'pi', x: 4.4, y: 2.4, shape: 'circle', w: 1, h: 1, label: '$\\pi^{(k)}$', tone: 'ink' },
  ],
  edges: [
    { from: 'p', to: 't', route: 'straight' },
    { from: 't', to: 'n', route: 'straight' },
    { from: 'pi', to: 'n', route: 'straight' },
  ],
  groups: [
    { id: 'N', label: 'items, $N$', tone: 'ink', rect: { x: 1.2, y: -1.1, w: 4.3, h: 2.1 }, labelAt: 'top-left' },
    {
      id: 'K',
      label: 'annotators, $K$',
      tone: 'ink',
      rect: { x: 3.5, y: -0.75, w: 3.8, h: 3.9 },
      labelAt: 'bottom-right',
    },
  ],
}

/** The Dawid–Skene model as a plate diagram with crossed item and annotator plates. */
export function DawidSkenePlate() {
  return (
    <Interactive
      title="The Dawid–Skene model"
      caption="Each item's true class t_i is drawn from the class prior p and is never observed. Annotator k's responses to item i (shaded: how often each class was given) depend on the true class through that annotator's confusion matrix π⁽ᵏ⁾. The responses sit where the item plate and the annotator plate cross: one set per item–annotator pair."
    >
      <Diagram
        spec={spec}
        ariaLabel="Plate diagram: class prior p points to the latent true class t_i in the item plate; t_i and the annotator's confusion matrix pi^(k) point to the observed responses, which lie in both the item and the annotator plates"
      />
    </Interactive>
  )
}
