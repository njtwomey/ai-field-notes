import type { Specimen } from '../../../specimen'
import { HeatSpecimen, TransportSchemesSpecimen, WaveSpecimen } from './_pde/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/dynamics/pde',
    title: 'The heat equation and its stability limit',
    description:
      'Explicit, implicit and Crank–Nicolson steps of u_t = u_xx, with the diffusion number r against its limit.',
    tags: ['heat equation', 'method of lines', 'CFL', 'Crank–Nicolson', 'stability'],
    render: () => <HeatSpecimen />,
  },
  {
    module: 'applied/dynamics/pde',
    title: 'Transport schemes after whole periods',
    description:
      'Upwind, Lax–Friedrichs and Lax–Wendroff on u_t + u_x = 0: numerical diffusion, dispersion and the CFL condition.',
    tags: ['advection', 'upwind', 'Lax–Wendroff', 'CFL', 'numerical diffusion'],
    render: () => <TransportSchemesSpecimen />,
  },
  {
    module: 'applied/dynamics/pde',
    title: 'The wave equation by leapfrog',
    description: 'A plucked string on a grid, with the Courant number and the discrete energy.',
    tags: ['wave equation', 'leapfrog', 'CFL', 'energy'],
    render: () => <WaveSpecimen />,
  },
]
