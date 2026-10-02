import type { Specimen } from '../../specimen'
import {
  ConformalRegressionSpecimen,
  ConformalSetsSpecimen,
  CoverageLawSpecimen,
  ReliabilityMapsSpecimen,
} from './_calibration/figures'

export const specimens: Specimen[] = [
  {
    module: 'learning/calibration',
    title: 'Calibration and conformal prediction',
    description:
      'Reliability diagrams before and after temperature, Dirichlet, histogram, isotonic, beta and Platt maps; split conformal and CQR intervals against the target coverage; the Beta law of coverage; LAC, APS and RAPS sets and their sizes.',
    tags: [
      'calibration',
      'reliability diagram',
      'ECE',
      'temperature scaling',
      'Dirichlet calibration',
      'isotonic',
      'conformal prediction',
      'CQR',
      'APS',
      'coverage',
    ],
    render: () => (
      <>
        <ReliabilityMapsSpecimen />
        <ConformalRegressionSpecimen />
        <CoverageLawSpecimen />
        <ConformalSetsSpecimen />
      </>
    ),
  },
]
