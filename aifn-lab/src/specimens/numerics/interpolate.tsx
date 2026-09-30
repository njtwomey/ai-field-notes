import type { Specimen } from '../../specimen'
import { InterpolantsCompared, PsplineSmoothing } from './_interpolate/figures'

export const specimens: Specimen[] = [
  {
    module: 'numerics/interpolate',
    title: 'Spline interpolants compared',
    description:
      'Linear, natural and not-a-knot cubic splines, PCHIP, Akima and the interpolating polynomial through draggable points, with the second derivative of one.',
    tags: ['cubicSpline', 'pchip', 'akima', 'interpolatingPolynomial', 'handles'],
    render: () => <InterpolantsCompared />,
  },
  {
    module: 'numerics/interpolate',
    title: 'P-spline smoothing against λ',
    description: 'pspline with its ± 2 se band, the GCV curve over log λ (draggable) and the scaled B-spline basis.',
    tags: ['pspline', 'psplineGcvPath', 'bsplineBasis', 'GCV', 'effective degrees of freedom'],
    render: () => <PsplineSmoothing />,
  },
]
