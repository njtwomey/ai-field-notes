import type { Specimen } from '@lab/specimen'
import { BifurcationSpecimen, PlaneMapSpecimen } from './_maps/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/differential-equations',
    title: 'Bifurcation diagram and cobweb',
    description:
      'The attractors of the logistic, sine and tent maps against their parameter, with a draggable r, the cobweb and the Lyapunov exponent.',
    tags: ['bifurcation', 'period doubling', 'cobweb', 'Lyapunov exponent', 'logistic map', 'chaos'],
    render: () => <BifurcationSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/differential-equations',
    title: 'Chaotic maps of the plane',
    description: 'The Hénon attractor and the standard map’s phase space, with their Lyapunov spectra.',
    tags: ['Hénon map', 'standard map', 'strange attractor', 'Lyapunov spectrum'],
    render: () => <PlaneMapSpecimen />,
  },
]
