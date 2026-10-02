import type { Specimen } from '../../specimen'
import { ChirpSpectrogramSpecimen } from './_shared/figures'
import { CoherenceSpecimen } from './_spectral/coherence'
import { SpectralEstimationSpecimen } from './_spectral/estimation'
import { LombScargleSpecimen } from './_spectral/lomb-scargle'

export const specimens: Specimen[] = [
  {
    module: 'signal/spectral',
    title: 'Spectrogram of a chirp',
    description: 'A noisy linear chirp: its spectrogram as the segment length and window change, and its Welch PSD.',
    tags: ['STFT', 'spectrogram', 'Welch', 'window', 'chirp'],
    render: () => <ChirpSpectrogramSpecimen />,
  },
  {
    module: 'signal/spectral',
    title: 'Spectral estimation: estimates against the true spectrum',
    description:
      'Periodogram, Welch, Bartlett, Blackman–Tukey, multitaper, AR and MUSIC estimates of AR processes and sinusoids in noise, overlaid on the exact PSD with χ² bands, bias and variance.',
    tags: [
      'periodogram',
      'Welch',
      'Bartlett',
      'Blackman–Tukey',
      'multitaper',
      'Burg',
      'MUSIC',
      'ESPRIT',
      'bias–variance',
      'leakage',
    ],
    render: () => <SpectralEstimationSpecimen />,
  },
  {
    module: 'signal/spectral',
    title: 'Lomb–Scargle: spectra from uneven sampling',
    description:
      'An unevenly sampled sinusoid: Lomb–Scargle with its false-alarm level against interpolated and zero-filled periodograms, and the spectral window that creates aliases.',
    tags: ['Lomb–Scargle', 'uneven sampling', 'aliasing', 'false-alarm probability', 'spectral window'],
    render: () => <LombScargleSpecimen />,
  },
  {
    module: 'signal/spectral',
    title: 'Coherence between two signals',
    description:
      'A delayed, filtered copy of an AR process plus noise: cross-spectrum magnitude and phase and the magnitude-squared coherence, estimated by Welch averaging, against the truth.',
    tags: ['coherence', 'cross-spectrum', 'csd', 'Welch', 'delay'],
    render: () => <CoherenceSpecimen />,
  },
]
