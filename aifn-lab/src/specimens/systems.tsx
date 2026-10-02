import type { Specimen } from '../specimen'
import { BodeMarginsSpecimen } from './_systems/figures'
import { IdentificationSpecimen } from './_systems/identification'
import { LoopShapingSpecimen } from './_systems/loop-shaping'

export const specimens: Specimen[] = [
  {
    module: 'systems',
    title: 'Bode plot and margins',
    description: 'Magnitude and phase of an open loop with its gain, phase and delay margins marked.',
    tags: ['Bode', 'frequency response', 'gain margin', 'phase margin', 'delay'],
    render: () => <BodeMarginsSpecimen />,
  },
  {
    module: 'systems',
    title: 'Root locus, Bode and Nyquist',
    description:
      'One open loop with draggable poles and zeros read three ways: the root locus with the closed-loop poles at a gain, the Bode plot with margins, and the Nyquist plot with its encirclements of −1; Routh–Hurwitz agrees.',
    tags: ['root locus', 'Nyquist', 'Bode', 'Routh–Hurwitz', 'gain margin', 'phase margin', 'poles and zeros'],
    render: () => <LoopShapingSpecimen />,
  },
  {
    module: 'systems',
    title: 'System identification',
    description:
      'A known second-order system recovered from noisy data by ARX (least squares), ARMAX (prediction-error method) and N4SID (subspace), with order selection by AIC and by singular values.',
    tags: ['system identification', 'ARX', 'ARMAX', 'prediction-error method', 'N4SID', 'subspace', 'model order'],
    render: () => <IdentificationSpecimen />,
  },
]
