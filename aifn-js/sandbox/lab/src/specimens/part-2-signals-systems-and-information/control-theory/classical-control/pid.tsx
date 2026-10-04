import type { Specimen } from '@lab/specimen'
import { PidSpecimen } from './_control/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/control-theory/classical-control',
    title: 'PID control',
    description: 'A PID loop around a third-order lag with actuator limits and anti-windup.',
    tags: ['PID', 'anti-windup', 'saturation', 'step response', 'trace'],
    render: () => <PidSpecimen />,
  },
]
