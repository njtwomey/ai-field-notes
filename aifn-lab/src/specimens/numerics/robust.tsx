import type { Specimen } from '../../specimen'
import { TwoViewRansacSpecimen } from './_robust/two-view'

export const specimens: Specimen[] = [
  {
    module: 'numerics/robust',
    title: 'Two-view geometry and RANSAC',
    description:
      'Synthetic correspondences between two cameras with a chosen share of outliers: RANSAC fits a homography (plane) or a fundamental matrix (depth) sample by sample; a draggable probe shows its mapped point or epipolar line, estimated against true.',
    tags: ['RANSAC', 'homography', 'fundamental matrix', 'epipolar geometry', 'eight-point algorithm', 'outliers'],
    render: () => <TwoViewRansacSpecimen />,
  },
]
