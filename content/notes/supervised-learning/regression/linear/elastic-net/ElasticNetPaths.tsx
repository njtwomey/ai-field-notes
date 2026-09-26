import { useState } from 'react'
import { Interactive, ParamChoice, Readout } from '@/components/viz'
import type { ElasticNetPaths as Paths } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { CoefficientPathExplorer } from '../_shared/CoefficientPathExplorer'

/** Elastic-net paths on the diabetes data for several l1 ratios, precomputed by python/mlc/figures/linear_models.py. */
export function ElasticNetPaths() {
  const { data } = useFigure<Paths>('elastic-net/paths')
  const [choice, setChoice] = useState('0.5')
  if (!data) return null
  const k = data.l1_ratios.findIndex((r) => String(r) === choice)
  const path = data.paths[k]
  return (
    <Interactive
      title="Elastic-net paths"
      caption="Elastic-net coefficients on the diabetes data for four mixing ratios; ratio 1 is the lasso. Drag the α line or use the slider. As the ratio falls, the path is smoother, more coefficients are nonzero at any α, and the correlated pair s1 and s2 share weight. At ratio 0.1 they enter within a factor of 1.3 in α and keep coefficients of the same sign and similar size; the lasso lets s1 in at α ≈ 3.2 and holds s2 at zero until α ≈ 0.25."
      controls={
        <ParamChoice
          label="l1 ratio"
          value={choice}
          onChange={setChoice}
          options={data.l1_ratios.map((r) => ({ value: String(r), label: r === 1 ? '1 (lasso)' : String(r) }))}
        />
      }
    >
      <CoefficientPathExplorer key={choice} path={path} symbol="α" start={1}>
        {(i) => (
          <Readout
            label="nonzero coefficients"
            value={`${path.coef.filter((row) => row[i] !== 0).length} of ${path.features.length}`}
          />
        )}
      </CoefficientPathExplorer>
    </Interactive>
  )
}
