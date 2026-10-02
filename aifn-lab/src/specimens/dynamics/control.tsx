import type { Specimen } from '../../specimen'
import { LqrCartPoleSpecimen, LqrPolesSpecimen, PolePlacementSpecimen } from './_control/figures'
import { LqgSpecimen } from './_control/lqg'
import { MpcSpecimen } from './_control/mpc'

export const specimens: Specimen[] = [
  {
    module: 'dynamics/control',
    title: 'LQR on a cart-pole',
    description:
      'The linear-quadratic regulator for the linearised cart-pole: state and angle weights against force weight, with the trajectories and control effort it gives.',
    tags: ['LQR', 'Riccati', 'Kleinman', 'state feedback', 'cart-pole'],
    render: () => <LqrCartPoleSpecimen />,
  },
  {
    module: 'dynamics/control',
    title: 'LQR poles',
    description:
      'The closed-loop poles of the cart-pole LQR as the force weight R sweeps six decades: the symmetric root locus.',
    tags: ['LQR', 'poles', 'root locus', 'Riccati'],
    render: () => <LqrPolesSpecimen />,
  },
  {
    module: 'dynamics/control',
    title: 'Pole placement',
    description:
      "Ackermann's formula on the double integrator: drag the closed-loop poles and see the gain and response.",
    tags: ['pole placement', 'Ackermann', 'state feedback', 'double integrator'],
    render: () => <PolePlacementSpecimen />,
  },
  {
    module: 'dynamics/control',
    title: 'Control: from LQR to MPC',
    description:
      'The double integrator with bounded force and velocity: LQR with its force clipped against receding-horizon MPC, the plan at each step drawn over what was applied; a horizon slider, a draggable start and soft (slack) velocity bounds.',
    tags: [
      'MPC',
      'model predictive control',
      'receding horizon',
      'LQR',
      'constraints',
      'quadratic program',
      'double integrator',
      'soft constraints',
    ],
    render: () => <MpcSpecimen />,
  },
  {
    module: 'dynamics/control',
    title: 'LQG: estimate, then control',
    description:
      'The noisy double integrator with only its position measured: the Kalman filter estimates the state, LQR acts on the estimate (certainty equivalence), against full-state LQR on the same noise; noise sliders, a mistuned filter and the separated closed-loop poles.',
    tags: ['LQG', 'Kalman filter', 'LQR', 'separation principle', 'certainty equivalence', 'output feedback'],
    render: () => <LqgSpecimen />,
  },
]
