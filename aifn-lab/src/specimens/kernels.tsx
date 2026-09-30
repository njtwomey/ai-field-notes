import type { Specimen } from '../specimen'
import { KernelGallery } from './_kernels/figures'

export const specimens: Specimen[] = [
  {
    module: 'kernels',
    title: 'Kernel gallery',
    description:
      'RBF, Matérn ½, 3⁄2, 5⁄2, rational quadratic, periodic, linear and cubic kernels: k(x, x₀) against a draggable x₀, prior draws and the Gram matrix, by lengthscale ℓ.',
    tags: ['rbf', 'matern', 'periodic', 'gram', 'lengthscale', 'samplePrior'],
    render: () => <KernelGallery />,
  },
]
