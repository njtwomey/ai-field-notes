import type { Specimen } from '../../specimen'
import { BitsErrorSpecimen, ServingCostSpecimen, TrainedMlpSpecimen } from './_quantise/figures'
import { QatSpecimen } from './_quantise/qat'

export const specimens: Specimen[] = [
  {
    module: 'nn/quantise',
    title: 'Quantisation: bits against error',
    description:
      'A weight histogram with its quantisation levels, error against bits for per-tensor, per-channel and stochastic rounding; a small MLP trained in the browser and quantised by round-to-nearest, GPTQ and AWQ; and the serving memory and decoding speed fewer bits buy.',
    tags: ['quantisation', 'int8', 'per-channel', 'stochastic rounding', 'GPTQ', 'AWQ', 'SQNR', 'roofline', 'KV cache'],
    render: () => (
      <>
        <BitsErrorSpecimen />
        <TrainedMlpSpecimen />
        <ServingCostSpecimen />
      </>
    ),
  },
  {
    module: 'nn/quantise',
    title: 'Quantisation-aware training',
    description:
      'A small MLP trained in the browser, then quantised at 2 to 6 bits by rounding once (PTQ) or by fine-tuning through fake-quantised weights with the straight-through estimator (QAT): accuracy against bits, the quantised loss while fine-tuning, and the weights against the integer levels.',
    tags: ['quantisation', 'QAT', 'PTQ', 'fake quantisation', 'straight-through estimator', 'fine-tuning'],
    render: () => <QatSpecimen />,
  },
]
