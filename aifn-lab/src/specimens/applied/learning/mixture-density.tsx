import type { Specimen } from '../../../specimen'
import { ArmShowcase } from './_mixture-density/showcase-arm'
import { InverseShowcase } from './_mixture-density/showcase-inverse'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/mixture-density',
    title: 'Showcase: mixture density networks',
    description:
      'A squared-error MLP against a mixture density network (Bishop, 1994), trained side by side in the browser on two inverse problems: the folded sine x = t + 0.3 sin 2πt, where the conditional mean cuts through the gaps and the MDN finds every branch, with a draggable slice of p(t | x); and two-link arm kinematics, where the MDN proposes both elbow bends and the squared-error network an arm that reaches neither.',
    tags: [
      'showcase',
      'mixture density network',
      'MDN',
      'inverse problem',
      'multimodal regression',
      'inverse kinematics',
      'mixtureDensityRun',
      'mixtureDensityNll',
    ],
    render: () => (
      <>
        <InverseShowcase />
        <ArmShowcase />
      </>
    ),
  },
]
