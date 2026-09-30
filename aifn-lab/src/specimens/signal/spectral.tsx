import type { Specimen } from '../../specimen'
import { ChirpSpectrogramSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'signal/spectral',
    title: 'Spectrogram of a chirp',
    description: 'A noisy linear chirp: its spectrogram as the segment length and window change, and its Welch PSD.',
    tags: ['STFT', 'spectrogram', 'Welch', 'window', 'chirp'],
    render: () => <ChirpSpectrogramSpecimen />,
  },
]
