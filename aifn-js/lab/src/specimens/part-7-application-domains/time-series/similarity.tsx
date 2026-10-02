import type { Specimen } from '@lab/specimen'
import { DtwFigure, MatrixProfileFigure } from './_similarity/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-7-application-domains/time-series',
    title: 'Time-series similarity: matrix profile and DTW',
    description:
      'A series with a planted motif and a discord: its matrix profile computed anytime by SCRIMP++ and played diagonal batch by batch against the exact STOMP profile, with the MASS distance profile of a draggable window; and dynamic time warping between two warped versions of one shape, the cost table filled row by row inside a Sakoe–Chiba band, the warping path, and LB_Keogh and LB_Kim.',
    tags: [
      'matrix profile',
      'SCRIMP++',
      'STOMP',
      'MASS',
      'motif',
      'discord',
      'DTW',
      'Sakoe–Chiba',
      'LB_Keogh',
      'scrimpSteps',
      'dtwProgram',
    ],
    render: () => (
      <>
        <MatrixProfileFigure />
        <DtwFigure />
      </>
    ),
  },
]
