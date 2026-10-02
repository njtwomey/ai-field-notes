import type { Specimen } from '../../specimen'
import { CannySpecimen, FeaturesSpecimen, MorphologySpecimen } from './_image/figures'

export const specimens: Specimen[] = [
  {
    module: 'signal/image',
    title: 'Classical vision: edges, corners and lines',
    description:
      'A synthetic scene with known geometry through Canny (every stage, scored against the true edges), Harris and Shi–Tomasi corners, Hough lines and circles, morphology and the Laplacian pyramid.',
    tags: ['Canny', 'edge detection', 'Harris', 'Shi–Tomasi', 'Hough transform', 'morphology', 'Laplacian pyramid'],
    render: () => (
      <>
        <CannySpecimen />
        <FeaturesSpecimen />
        <MorphologySpecimen />
      </>
    ),
  },
]
