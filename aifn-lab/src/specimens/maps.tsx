import type { Specimen } from '../specimen'
import { BifurcationSpecimen, PlaneMapSpecimen } from './_maps/figures'

export const specimens: Specimen[] = [
  {
    module: 'maps',
    title: 'Bifurcation diagram and cobweb',
    description:
      'The attractors of the logistic, sine and tent maps against their parameter, with a draggable r, the cobweb and the Lyapunov exponent.',
    tags: ['bifurcation', 'period doubling', 'cobweb', 'Lyapunov exponent', 'logistic map', 'chaos'],
    render: () => <BifurcationSpecimen />,
  },
  {
    module: 'maps',
    title: 'Chaotic maps of the plane',
    description: 'The Hénon attractor and the standard map’s phase space, with their Lyapunov spectra.',
    tags: ['Hénon map', 'standard map', 'strange attractor', 'Lyapunov spectrum'],
    render: () => <PlaneMapSpecimen />,
  },
]
