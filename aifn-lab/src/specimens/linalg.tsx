import type { Specimen } from '../specimen'
import { CholeskyJitterSpecimen, EighSpecimen, LuSpecimen, QrSpecimen, SvdSpecimen } from './_linalg/figures'

export const specimens: Specimen[] = [
  {
    module: 'linalg',
    title: 'Cholesky with jitter reporting',
    description:
      'The Cholesky factor of a squared-exponential Gram matrix. As ℓ grows the matrix becomes numerically singular; the smallest jitter that lets it factor is added and reported.',
    tags: ['cholesky', 'jitter', 'gaussian processes', 'kernel'],
    render: () => <CholeskyJitterSpecimen />,
  },
  {
    module: 'linalg',
    title: 'LU with partial pivoting',
    description:
      'PA = LU of a matrix that is singular at ε = 0, where `singular` is reported instead of dividing by zero.',
    tags: ['lu', 'pivoting', 'singular'],
    render: () => <LuSpecimen />,
  },
  {
    module: 'linalg',
    title: 'Symmetric eigendecomposition',
    description: 'Jacobi eigh of a Gram matrix: eigenvalues descending on a log scale, and the leading eigenvectors.',
    tags: ['eigh', 'jacobi', 'spectrum'],
    render: () => <EighSpecimen />,
  },
  {
    module: 'linalg',
    title: 'Truncated SVD',
    description: 'The rank-k reconstruction of a small pattern from its one-sided Jacobi SVD, and its singular values.',
    tags: ['svd', 'low rank', 'Eckart–Young'],
    render: () => <SvdSpecimen />,
  },
  {
    module: 'linalg',
    title: 'Householder QR',
    description: 'A tall matrix, its orthonormal Q and upper-triangular R, and QᵀQ = I.',
    tags: ['qr', 'householder', 'orthogonal'],
    render: () => <QrSpecimen />,
  },
]
