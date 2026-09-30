import type { Specimen } from '../specimen'
import { DecimationSpecimen, EllipseSpecimen, HullContourSpecimen } from './_geometry/figures'

export const specimens: Specimen[] = [
  {
    module: 'geometry',
    title: 'Covariance ellipses',
    description: 'Gaussian level sets at 50%, 90% and 99% probability mass, with a draggable mean.',
    tags: ['ellipse', 'covariance', 'Mahalanobis'],
    render: () => <EllipseSpecimen />,
  },
  {
    module: 'geometry',
    title: 'Hulls and contours',
    description: 'A convex hull of a point cloud and contour lines of a field by marching squares.',
    tags: ['convex hull', 'marching squares', 'contours'],
    render: () => <HullContourSpecimen />,
  },
  {
    module: 'geometry',
    title: 'Decimation for drawing',
    description: 'LTTB, min–max and stride decimation of a 5000-point line.',
    tags: ['LTTB', 'decimation', 'downsampling'],
    render: () => <DecimationSpecimen />,
  },
]
