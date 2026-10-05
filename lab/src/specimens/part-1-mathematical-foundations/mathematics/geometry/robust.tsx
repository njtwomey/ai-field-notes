import type { Specimen } from '@lab/specimen'
import { TwoViewRansacSpecimen } from './_robust/two-view'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/geometry',
    title: 'Two-view geometry and RANSAC',
    description:
      'Synthetic correspondences between two cameras with a chosen share of outliers: RANSAC fits a homography (plane) or a fundamental matrix (depth) sample by sample; a draggable probe shows its mapped point or epipolar line, estimated against true.',
    tags: ['RANSAC', 'homography', 'fundamental matrix', 'epipolar geometry', 'eight-point algorithm', 'outliers'],
    render: () => <TwoViewRansacSpecimen />,
  },
]
