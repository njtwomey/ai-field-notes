import type { Specimen } from '../specimen'
import { ChirpSpectrogramSpecimen, FilterSpecimen, ScalogramSpecimen, SiftSpecimen } from './_dsp/figures'

export const specimens: Specimen[] = [
  {
    module: 'dsp',
    title: 'Spectrogram of a chirp',
    description: 'A noisy linear chirp: its spectrogram as the segment length and window change, and its Welch PSD.',
    tags: ['STFT', 'spectrogram', 'Welch', 'window', 'chirp'],
    render: () => <ChirpSpectrogramSpecimen />,
  },
  {
    module: 'dsp',
    title: 'Filter responses and filtering',
    description:
      'Butterworth, Chebyshev and FIR low-pass filters: magnitude responses and causal against zero-phase filtering.',
    tags: ['Butterworth', 'Chebyshev', 'FIR', 'lfilter', 'filtfilt', 'freqz'],
    render: () => <FilterSpecimen />,
  },
  {
    module: 'dsp',
    title: 'Wavelet scalogram',
    description: 'The Morlet continuous wavelet transform of a frequency switch, a chirp and an impulse.',
    tags: ['wavelet', 'CWT', 'Morlet', 'time–frequency'],
    render: () => <ScalogramSpecimen />,
  },
  {
    module: 'dsp',
    title: 'Empirical mode decomposition',
    description: 'Sifting one intrinsic mode function, step by step, with its envelopes.',
    tags: ['EMD', 'sifting', 'trace', 'Hilbert–Huang'],
    render: () => <SiftSpecimen />,
  },
]
