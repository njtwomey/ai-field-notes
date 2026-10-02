import type { Specimen } from '@lab/specimen'
import { FilterSpecimen } from '../_shared/figures'
import { DenoisingFigure, MatchedFilterFigure } from '../_transforms/denoising'
import { ApplyFilterFigure, FamiliesFigure, FilterGalleryFigure } from '../_transforms/filter-gallery'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Filter responses and filtering',
    description:
      'Butterworth, Chebyshev and FIR low-pass filters: magnitude responses and causal against zero-phase filtering.',
    tags: ['Butterworth', 'Chebyshev', 'FIR', 'lfilter', 'filtfilt', 'freqz'],
    render: () => <FilterSpecimen />,
  },
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Filter design gallery',
    description:
      'Every registered design family (Butterworth, Chebyshev I and II, elliptic, Bessel, FIR window method, Parks–McClellan) in any band type: magnitude, phase, group delay, impulse and step responses, and a pole–zero plot whose poles and zeros can be dragged; the IIR families compared at equal order; a design applied causally and with zero phase.',
    tags: [
      'filter design',
      'Butterworth',
      'Chebyshev',
      'elliptic',
      'Bessel',
      'Parks–McClellan',
      'remez',
      'firwin',
      'poles and zeros',
      'group delay',
      'filtfilt',
      'ellip',
      'bessel',
    ],
    render: () => (
      <>
        <FilterGalleryFigure />
        <FamiliesFigure />
        <ApplyFilterFigure />
      </>
    ),
  },
  {
    module: 'part-2-signals-systems-and-information/signal-processing/filtering',
    title: 'Denoising: Wiener, Savitzky–Golay, median, wavelet thresholding',
    description:
      'The Donoho–Johnstone test functions with Gaussian noise and impulses, cleaned by frequency-domain and local Wiener filters, Savitzky–Golay smoothing, the running median and wavelet shrinkage, each scored by SNR against the known clean signal; and the matched filter detecting a known pulse.',
    tags: [
      'denoising',
      'Wiener filter',
      'Savitzky–Golay',
      'median filter',
      'wavelet shrinkage',
      'matched filter',
      'SNR',
      'savgolFilter',
      'medfilt',
      'wienerDenoise',
      'waveletDenoise',
      'matchedFilter',
    ],
    render: () => (
      <>
        <DenoisingFigure />
        <MatchedFilterFigure />
      </>
    ),
  },
]
