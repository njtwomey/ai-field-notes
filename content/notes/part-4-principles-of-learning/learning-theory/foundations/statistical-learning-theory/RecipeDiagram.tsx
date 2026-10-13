import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const STEP = 1.9
const BOX = { x: 0, w: 7.4, h: 1.45 }
const TEXT = { x: 8.6, w: 6.4, shape: 'text' as const, small: true }

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    {
      id: 'decompose',
      ...BOX,
      y: 0,
      tone: 0,
      label: '1 · Decompose\n$R(\\hat h) - R^* \\le \\text{approx.} + 2\\sup_{h}\\lvert R(h) - \\hat R_S(h)\\rvert$',
    },
    {
      id: 'concentrate',
      ...BOX,
      y: STEP,
      tone: 1,
      label:
        '2 · Concentrate (one $h$)\n$\\prob(\\lvert \\hat R_S(h) - R(h)\\rvert \\ge \\epsilon) \\le 2e^{-2m\\epsilon^2}$',
    },
    {
      id: 'uniform',
      ...BOX,
      y: 2 * STEP,
      tone: 2,
      label: '3 · Make it uniform (all $h \\in \\Hcal$)\nunion bound, symmetrisation, capacity of $\\Hcal$',
    },
    {
      id: 'structure',
      ...BOX,
      y: 3 * STEP,
      tone: 3,
      label: '4 · Exploit structure\nmargins, norms, stability, priors, low noise',
    },
    {
      id: 'invert',
      ...BOX,
      y: 4 * STEP,
      tone: 'ink',
      label: '5 · Invert and read\nsolve for $\\epsilon$ at $\\delta$, or for $m$; compare with a test error',
    },
    { id: 't1', ...TEXT, y: 0, label: 'foundations: ERM, the gap, the three-way trade-off' },
    { id: 't2', ...TEXT, y: STEP, label: 'Hoeffding, Bernstein, binomial tail: test-set bounds' },
    { id: 't3', ...TEXT, y: 2 * STEP, label: 'PAC learning; uniform convergence: VC, Rademacher, covering numbers' },
    { id: 't4', ...TEXT, y: 3 * STEP, label: 'margins and norms; stability, PAC-Bayes, compression; fast rates' },
    { id: 't5', ...TEXT, y: 4 * STEP, label: 'sample complexity; is the bound vacuous?' },
  ],
  edges: [
    { from: 'decompose:s', to: 'concentrate:n' },
    { from: 'concentrate:s', to: 'uniform:n' },
    { from: 'uniform:s', to: 'structure:n' },
    { from: 'structure:s', to: 'invert:n' },
  ],
}

/** The five moves that every generalisation bound in the chapter makes, with the sub-chapters that make each one. */
export function RecipeDiagram() {
  return (
    <Figure
      title="The recipe for a generalisation bound"
      caption="Every bound in the chapter makes these moves in order. Move 1 reduces the excess risk to a uniform deviation over the class. Move 2 controls the deviation of one fixed hypothesis. Move 3 pays for the size of the class, by a union bound for finite classes and by symmetrisation and a capacity measure for infinite ones. Move 4 replaces the crude capacity by something finer: a margin, a norm, the algorithm's stability or a prior. Move 5 turns the probability statement into an accuracy, or into the sample size an accuracy needs. The right column names where each move is developed."
    >
      <Diagram
        spec={spec}
        ariaLabel="Five steps in a column: decompose the excess risk into approximation error and twice the uniform deviation; concentrate the empirical risk of one hypothesis; make the bound uniform over the class; exploit structure such as margins, stability or priors; invert the bound for the accuracy or the sample size"
      />
    </Figure>
  )
}
