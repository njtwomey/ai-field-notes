import type { Specimen } from '@lab/specimen'
import { CapacitySpecimen, RateDistortionSpecimen } from './_channels/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/information-theory/channel-capacity-and-coding',
    title: 'Blahut–Arimoto converging to capacity',
    description:
      'The traced capacity algorithm on five channels: upper and lower bounds meet at the capacity, which matches the closed forms for the binary symmetric, erasure and Z channels.',
    tags: ['blahutArimotoCapacity', 'channelCapacity', 'trace', 'channel'],
    render: () => <CapacitySpecimen />,
  },
  {
    module: 'part-2-signals-systems-and-information/information-theory/channel-capacity-and-coding',
    title: 'Rate–distortion curve',
    description:
      'rateDistortionCurve for a binary source with Hamming distortion, one Blahut–Arimoto run per slope, against R(D) = H(p) − H(D).',
    tags: ['rateDistortion', 'Blahut–Arimoto', 'R(D)'],
    render: () => <RateDistortionSpecimen />,
  },
]
