import type { Specimen } from '@lab/specimen'
import { CepstrumPitchFigure, PitchTrackFigure } from '../_transforms/pitch'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/spectral-estimation',
    title: 'Pitch and the cepstrum',
    description:
      'A synthetic voiced sound with a known f₀: its harmonic log spectrum and liftered formant envelope, the real cepstrum with its peak at one period, and the YIN difference function with a draggable threshold; pitch tracks against the truth under vibrato.',
    tags: [
      'cepstrum',
      'pitch',
      'YIN',
      'liftering',
      'formants',
      'f0',
      'realCepstrum',
      'cepstralPitch',
      'yinPitch',
      'cepstralEnvelope',
    ],
    render: () => (
      <>
        <CepstrumPitchFigure />
        <PitchTrackFigure />
      </>
    ),
  },
]
