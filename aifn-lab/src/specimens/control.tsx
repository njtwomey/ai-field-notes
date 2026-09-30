import type { Specimen } from '../specimen'
import {
  BodeMarginsSpecimen,
  LqrCartPoleSpecimen,
  LqrPolesSpecimen,
  PidSpecimen,
  PolePlacementSpecimen,
} from './_control/figures'

export const specimens: Specimen[] = [
  {
    module: 'control',
    title: 'LQR on a cart-pole',
    description:
      'The linear-quadratic regulator for the linearised cart-pole: state and angle weights against force weight, with the trajectories and control effort it gives.',
    tags: ['LQR', 'Riccati', 'Kleinman', 'state feedback', 'cart-pole'],
    render: () => <LqrCartPoleSpecimen />,
  },
  {
    module: 'control',
    title: 'LQR poles',
    description:
      'The closed-loop poles of the cart-pole LQR as the force weight R sweeps six decades: the symmetric root locus.',
    tags: ['LQR', 'poles', 'root locus', 'Riccati'],
    render: () => <LqrPolesSpecimen />,
  },
  {
    module: 'control',
    title: 'Pole placement',
    description:
      "Ackermann's formula on the double integrator: drag the closed-loop poles and see the gain and response.",
    tags: ['pole placement', 'Ackermann', 'state feedback', 'double integrator'],
    render: () => <PolePlacementSpecimen />,
  },
  {
    module: 'control',
    title: 'Bode plot and margins',
    description: 'Magnitude and phase of an open loop with its gain, phase and delay margins marked.',
    tags: ['Bode', 'frequency response', 'gain margin', 'phase margin', 'delay'],
    render: () => <BodeMarginsSpecimen />,
  },
  {
    module: 'control',
    title: 'PID control',
    description: 'A PID loop around a third-order lag with actuator limits and anti-windup.',
    tags: ['PID', 'anti-windup', 'saturation', 'step response', 'trace'],
    render: () => <PidSpecimen />,
  },
]
