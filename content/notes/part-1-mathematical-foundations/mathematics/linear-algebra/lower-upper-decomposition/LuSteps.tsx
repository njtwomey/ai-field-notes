import { useState } from 'react'
import { EquationSteps, Figure, type EquationStep } from 'aifn-render'

const A0 = String.raw`\begin{bmatrix} 4 & 3 & 0 \\ 3 & 4 & -1 \\ 0 & -1 & 4 \end{bmatrix}`
const A1 = String.raw`\begin{bmatrix} 4 & 3 & 0 \\ 0 & \tfrac{7}{4} & -1 \\ 0 & -1 & 4 \end{bmatrix}`
const U = String.raw`\begin{bmatrix} 4 & 3 & 0 \\ 0 & \tfrac{7}{4} & -1 \\ 0 & 0 & \tfrac{24}{7} \end{bmatrix}`
// The multipliers fill L below the diagonal as elimination finds them; ∗ marks an entry not yet known.
const L1 = String.raw`\begin{bmatrix} 1 & 0 & 0 \\ \tfrac{3}{4} & 1 & 0 \\ \ast & \ast & 1 \end{bmatrix}`
const L2 = String.raw`\begin{bmatrix} 1 & 0 & 0 \\ \tfrac{3}{4} & 1 & 0 \\ 0 & \ast & 1 \end{bmatrix}`
const L = String.raw`\begin{bmatrix} 1 & 0 & 0 \\ \tfrac{3}{4} & 1 & 0 \\ 0 & -\tfrac{4}{7} & 1 \end{bmatrix}`

const STEPS: EquationStep[] = [
  {
    tex: String.raw`\Amat = ${A0}, \qquad \bvec = \begin{bmatrix} 10 \\ 8 \\ 10 \end{bmatrix}`,
    note: 'The system Ax = b to factorise and solve.',
  },
  {
    tex: String.raw`\ell_{21} = \frac{a_{21}}{a_{11}} = \frac{3}{4}, \qquad
      \text{row}_2 \leftarrow \text{row}_2 - \tfrac{3}{4}\,\text{row}_1: \quad ${A1}, \qquad \Lmat = ${L1}`,
    note: 'Clear entry (2, 1). The multiplier is the entry divided by the pivot 4, and it is stored in L at (2, 1).',
  },
  {
    tex: String.raw`\ell_{31} = \frac{a_{31}}{a_{11}} = \frac{0}{4} = 0, \qquad
      \text{row}_3 \text{ unchanged}, \qquad \Lmat = ${L2}`,
    note: 'Entry (3, 1) is already zero, so the multiplier is 0 and nothing is subtracted. Column 1 is done.',
  },
  {
    tex: String.raw`\ell_{32} = \frac{-1}{7/4} = -\frac{4}{7}, \qquad
      \text{row}_3 \leftarrow \text{row}_3 + \tfrac{4}{7}\,\text{row}_2: \quad \Umat = ${U}, \qquad \Lmat = ${L}`,
    note: 'Clear entry (3, 2) against the new pivot 7/4. The bottom-right entry becomes 4 − 4/7 = 24/7. The matrix is now upper triangular: it is U.',
  },
  {
    tex: String.raw`\Lmat\Umat = ${L}${U} = ${A0} = \Amat`,
    note: 'Multiplying the factors back gives A. For example, row 3 of LU is −4/7 times row 2 of U plus row 3 of U.',
  },
  {
    tex: String.raw`\Lmat\yvec = \bvec: \quad y_1 = 10`,
    note: 'Forward substitution, top row first. Row 1 of L is (1, 0, 0), so y₁ is read off.',
  },
  {
    tex: String.raw`y_2 = b_2 - \ell_{21}y_1 = 8 - \tfrac{3}{4}\cdot 10 = \tfrac{1}{2}`,
    note: 'Row 2 involves only y₁, which is known.',
  },
  {
    tex: String.raw`y_3 = b_3 - \ell_{31}y_1 - \ell_{32}y_2 = 10 - 0 + \tfrac{4}{7}\cdot\tfrac{1}{2} = \tfrac{72}{7}`,
    note: 'Row 3 involves y₁ and y₂. y is b after the same row operations as A.',
  },
  {
    tex: String.raw`\Umat\xvec = \yvec: \quad x_3 = \frac{y_3}{u_{33}} = \frac{72/7}{24/7} = 3`,
    note: 'Back substitution, bottom row first. Row 3 of U has one nonzero entry.',
  },
  {
    tex: String.raw`x_2 = \frac{y_2 - u_{23}x_3}{u_{22}} = \frac{\tfrac{1}{2} + 3}{7/4} = 2`,
    note: 'Row 2 involves x₂ and the known x₃.',
  },
  {
    tex: String.raw`x_1 = \frac{y_1 - u_{12}x_2 - u_{13}x_3}{u_{11}} = \frac{10 - 3\cdot 2 - 0}{4} = 1`,
    note: 'Row 1 gives the last unknown: x = (1, 2, 3).',
  },
  {
    tex: String.raw`\det\Amat = \det\Lmat\,\det\Umat = 1 \cdot \paren{4 \cdot \tfrac{7}{4} \cdot \tfrac{24}{7}} = 24`,
    note: 'L has a unit diagonal and U is triangular, so the determinant is the product of the pivots.',
  },
]

/** Gaussian elimination recorded as L and U, then the two triangular solves, one operation per step. */
export function LuSteps() {
  const [step, setStep] = useState(0)
  return (
    <Figure
      title="LU decomposition, one step at a time"
      hoverReadout={false}
      caption="Play or step through the factorisation of a 3 × 3 matrix and the solve that uses it. Each elimination step clears one entry below the diagonal and writes its multiplier into L; the ∗ entries of L are not yet known. When the matrix is upper triangular it is U. Forward substitution then solves Ly = b from the top, and back substitution solves Ux = y from the bottom. Earlier steps stay on screen, faded."
    >
      <EquationSteps steps={STEPS} step={step} onStep={setStep} />
    </Figure>
  )
}
