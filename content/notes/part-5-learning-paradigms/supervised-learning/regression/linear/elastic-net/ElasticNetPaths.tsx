import { useState } from 'react'
import { Choice, Figure, Readout, useComputed } from 'aifn-render'
import { CoefficientPathExplorer } from '../_shared/CoefficientPathExplorer'
import { elasticNetPathsData, L1_RATIOS } from '../_shared/diabetes'

/** Elastic-net paths on the diabetes data for several l1 ratios, from the engine's `elasticNetPath`. */
export function ElasticNetPaths() {
  const paths = useComputed(elasticNetPathsData, [], { mode: 'release' }).value
  const [ratio, setRatio] = useState('0.5')
  const path = paths[L1_RATIOS.findIndex((r) => String(r) === ratio)]
  return (
    <Figure
      title="Elastic-net paths"
      caption="Elastic-net coefficients on the diabetes data for four mixing ratios; ratio 1 is the lasso. Drag the α line or use the slider. As the ratio falls, the path is smoother, more coefficients are nonzero at any α, and the correlated pair s1 and s2 share weight. At ratio 0.1 they enter within a factor of 1.3 in α and keep coefficients of the same sign and similar size; the lasso lets s1 in at α ≈ 3.2 and holds s2 at zero until α ≈ 0.25."
      controls={
        <Choice
          label="l1 ratio"
          value={ratio}
          onChange={setRatio}
          options={L1_RATIOS.map((r) => ({ value: String(r), label: r === 1 ? '1 (lasso)' : String(r) }))}
        />
      }
    >
      <CoefficientPathExplorer key={ratio} path={path} symbol="α" start={1}>
        {(i) => <Readout label="nonzero coefficients" value={`${path.nonzero[i]} of ${path.features.length}`} />}
      </CoefficientPathExplorer>
    </Figure>
  )
}
