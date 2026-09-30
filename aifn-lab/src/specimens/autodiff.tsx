import type { Specimen } from '../specimen'
import { BackpropSpecimen } from './_autodiff/backprop'
import { DerivativeSpecimen, HvpSpecimen } from './_autodiff/figures'

export const specimens: Specimen[] = [
  {
    module: 'autodiff',
    title: 'A function and its derivatives',
    description:
      'f, f′ = grad(f) and f″ = grad(grad(f)) on a grid, from the primitives’ own derivative rules (log Γ goes through ψ and ψ₁), with the tangent at a draggable x₀ and a finite-difference gradCheck.',
    tags: ['grad', 'nested grad', 'second derivative', 'gradCheck'],
    render: () => <DerivativeSpecimen />,
  },
  {
    module: 'autodiff',
    title: 'Backpropagation, node by node',
    description:
      'traceGraph records a scalar function as a graph of primitives, drawn as a factor graph twice: the forward pass computes values left to right, the mirrored backward pass pulls adjoints ∂f/∂v right to left, each edge carrying its local partial. Step through both, one operation at a time.',
    tags: ['traceGraph', 'ComputationGraphView', 'factor graph', 'reverse mode', 'backpropagation', 'chain rule'],
    render: () => <BackpropSpecimen />,
  },
  {
    module: 'autodiff',
    title: 'Hessian-vector product on a quadratic',
    description:
      'For q(x) = ½ xᵀQx, hvp(q, x, v) equals Q·v at every x, as does jvp of grad q along v; hessian(q) recovers Q.',
    tags: ['hvp', 'hessian', 'jvp', 'quadratic'],
    render: () => <HvpSpecimen />,
  },
]
