import { Interactive, Readout } from '@/components/viz'
import type { LassoPath } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { CoefficientPathExplorer } from '../_shared/CoefficientPathExplorer'

/** Lasso coefficients on the diabetes data against α, precomputed by python/mlc/figures/linear_models.py. */
export function LassoPaths() {
  const { data } = useFigure<LassoPath>('lasso/path')
  if (!data) return null
  return (
    <Interactive
      title="The lasso path"
      caption="Lasso coefficients on the diabetes data as α falls from the value that zeroes every coefficient (right) to almost no penalty (left). Drag the α line or use the slider. Coefficients leave zero one at a time and the path is piecewise linear in α. Of the correlated pair s1 and s2, s1 enters at α ≈ 3.2 while s2 stays at exactly zero until α falls below 0.25."
    >
      <CoefficientPathExplorer path={data.path} symbol="α" ols={data.ols} start={1}>
        {(i) => <Readout label="nonzero coefficients" value={`${data.nonzero[i]} of ${data.path.features.length}`} />}
      </CoefficientPathExplorer>
    </Interactive>
  )
}
