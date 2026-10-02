import type { Specimen } from '../../specimen'
import { ClippingSpecimen, OffPolicySweepSpecimen, OneLogSpecimen } from './_off-policy/figures'

export const specimens: Specimen[] = [
  {
    module: 'learning/off-policy',
    title: 'Off-policy evaluation: IPS, DM and doubly robust',
    description:
      'Logged bandit feedback with a known true policy value: the bias and variance of IPS, clipped IPS, SNIPS, the direct method and doubly robust as logging propensities get extreme, the clipping trade-off, and one log’s importance weights.',
    tags: [
      'off-policy evaluation',
      'IPS',
      'SNIPS',
      'doubly robust',
      'direct method',
      'clipping',
      'propensity',
      'counterfactual',
    ],
    render: () => (
      <>
        <OffPolicySweepSpecimen />
        <ClippingSpecimen />
        <OneLogSpecimen />
      </>
    ),
  },
]
