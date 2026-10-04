import type { Specimen } from '@lab/specimen'
import { TradeOffFigure, WindowFigure } from '../_transforms/windows'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Windows and the leakage trade-off',
    description:
      "Every registered window's shape and spectrum, with its main-lobe width and peak side lobe measured against the registry's values, and the trade-off between them across the registry with a weak tone beside a strong one.",
    tags: [
      'window',
      'leakage',
      'side lobe',
      'main lobe',
      'Hann',
      'Hamming',
      'Blackman',
      'Kaiser',
      'windowRegistry',
      'getWindow',
    ],
    render: () => (
      <>
        <WindowFigure />
        <TradeOffFigure />
      </>
    ),
  },
]
