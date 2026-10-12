import { Figure, formatNumber, Readout, useComputed } from 'aifn-render'
import { CoefficientPathExplorer } from '../_shared/CoefficientPathExplorer'
import { ridgePathData } from '../_shared/diabetes'

/** Ridge coefficients on the diabetes data against λ, from the engine's `ridgePath`. */
export function RidgePaths() {
  const data = useComputed(ridgePathData, [], { mode: 'release' }).value
  return (
    <Figure
      title="Ridge shrinks every coefficient, none to zero"
      caption="Coefficients of a ridge fit to the diabetes data (ten standardised features, 442 patients) as λ grows from right to left. Drag the λ line or use the slider. Every coefficient shrinks smoothly towards zero and reaches it only as λ → ∞. The correlated pair s1 and s2, which OLS fits with large coefficients of opposite sign (−37.7 and +22.7), is tamed quickly: at λ = 100 both are small and negative. The effective degrees of freedom fall from 10 (OLS) towards 0."
    >
      <CoefficientPathExplorer path={data.path} symbol="λ" ols={data.ols} start={100}>
        {(i) => <Readout label="effective degrees of freedom" value={formatNumber(data.dof[i])} />}
      </CoefficientPathExplorer>
    </Figure>
  )
}
