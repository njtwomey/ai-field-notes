import type { Specimen } from '../specimen'
import { PhasePortraitSpecimen, TransportSpecimen } from './_fields/figures'

export const specimens: Specimen[] = [
  {
    module: 'fields',
    title: 'Phase portrait',
    description:
      'Direction field, nullclines, classified fixed points, saddle manifolds and a limit cycle of four planar systems, with a draggable start.',
    tags: ['phase portrait', 'fixed points', 'nullclines', 'stable manifold', 'limit cycle', 'Poincaré section'],
    render: () => <PhasePortraitSpecimen />,
  },
  {
    module: 'fields',
    title: 'Transport of a density along a flow',
    description: 'Liouville’s equation by characteristics, with particles pushed forward by the same flow.',
    tags: ['Liouville', 'continuity equation', 'divergence', 'push-forward', 'characteristics'],
    render: () => <TransportSpecimen />,
  },
]
