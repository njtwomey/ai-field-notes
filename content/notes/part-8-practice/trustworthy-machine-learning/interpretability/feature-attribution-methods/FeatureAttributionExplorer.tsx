import { useMemo, useState } from 'react'
import {
  Bars,
  ControlRow,
  Figure,
  Plot,
  Plots,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { normals, stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import {
  exactShapley,
  interventionalValue,
  kernelShap,
  lime,
  type ScalarModel,
} from 'aifn/learning/explain'

const METHODS = [
  { value: 'exactshap', label: 'Exact Shapley (all 2⁶ coalitions)' },
  { value: 'kernelshap', label: 'KernelSHAP (sampled linear surrogate)' },
  { value: 'lime', label: 'LIME (local exponential kernel surrogate)' },
]

export function FeatureAttributionExplorer() {
  const [methodChoice, setMethodChoice] = useState<'exactshap' | 'kernelshap' | 'lime'>('exactshap')
  const [sampleIdx, setSampleIdx] = useState(0)

  // Synthetic dataset of 30 test points in 6D
  const flatX = useMemo(() => {
    const s = stream('attribution-points')
    const X = normals(s, [30, 6])
    return Float64Array.from(toFlat(X))
  }, [])

  // Non-linear scoring model with main effects, interaction, and noise
  const scoringModel: ScalarModel = useMemo(() => {
    return (Xmat: Tensor) => {
      const [m, d] = Xmat.shape
      const flat = toFlat(Xmat)
      const out = new Float64Array(m)
      for (let r = 0; r < m; r++) {
        const x0 = flat[r * d + 0]
        const x1 = flat[r * d + 1]
        const x2 = flat[r * d + 2]
        const x3 = flat[r * d + 3]
        const x4 = flat[r * d + 4]
        const x5 = flat[r * d + 5]
        const z = 1.8 * x0 - 1.2 * x1 + 2.5 * x2 * x3 + 0.1 * x4 - 0.05 * x5
        out[r] = 1 / (1 + Math.exp(-z))
      }
      return out
    }
  }, [])

  // Background dataset (15 points)
  const background = useMemo(() => {
    const d = 6
    const bgN = 15
    return fromData(flatX.subarray(0, bgN * d), [bgN, d])
  }, [flatX])

  // Selected instance
  const instance = useMemo(() => {
    const d = 6
    const offset = sampleIdx * d
    return flatX.subarray(offset, offset + d)
  }, [flatX, sampleIdx])

  const prediction = useMemo(() => {
    const input = fromData(instance, [1, 6])
    const out = scoringModel(input)
    return toFlat(out as Tensor)[0]
  }, [scoringModel, instance])

  // Compute local attributions
  const { attributions, baseValue } = useMemo(() => {
    const d = 6
    let base = 0.5
    let values: number[] = new Array(d).fill(0)

    if (methodChoice === 'exactshap') {
      const v = interventionalValue(scoringModel, instance, background)
      const shap = exactShapley(v, d)
      values = Array.from(shap.values)
      base = shap.base
    } else if (methodChoice === 'kernelshap') {
      const kShap = kernelShap(scoringModel, instance, background, {
        samples: 60,
        stream: stream(`kernelshap-${sampleIdx}`),
      })
      values = Array.from(kShap.values)
      base = kShap.base
    } else {
      const s = stream(`lime-${sampleIdx}`)
      const l = lime(scoringModel, instance, s, { samples: 150 })
      values = Array.from(l.coefficients)
      base = l.intercept
    }

    return { attributions: values, baseValue: base }
  }, [scoringModel, instance, background, methodChoice, sampleIdx])

  // Global feature importance (mean absolute Exact Shapley over 10 instances)
  const globalImportance = useMemo(() => {
    const d = 6
    const nSamples = 10
    const imp = new Array(d).fill(0)

    for (let i = 0; i < nSamples; i++) {
      const inst = flatX.subarray(i * d, (i + 1) * d)
      const v = interventionalValue(scoringModel, inst, background)
      const shap = exactShapley(v, d)
      for (let j = 0; j < d; j++) {
        imp[j] += Math.abs(shap.values[j]) / nSamples
      }
    }
    return imp
  }, [scoringModel, flatX, background])

  const barIndices = [0, 1, 2, 3, 4, 5]

  const localX = useAxis({ label: 'Feature Index (0 to 5)', range: [-0.5, 5.5] })
  const localY = useAxis({ label: 'Attribution φⱼ', range: [-0.4, 0.4] })

  const globalX = useAxis({ label: 'Feature Index (0 to 5)', range: [-0.5, 5.5] })
  const globalY = useAxis({ label: 'Mean |Exact Shapley| Importance', range: [0, 0.3] })

  const sumAttr = attributions.reduce((a, b) => a + b, 0)

  return (
    <Figure
      title="Feature Attribution: Exact Shapley, KernelSHAP & LIME"
      purpose="Compare local feature attribution fidelity, non-linear feature interaction capture, and sampling variance across Shapley and surrogate methods."
      caption={
        'Interactive Feature Attribution comparison (Lundberg et al., 2020; Ribeiro et al., 2016). Features x₁ and x₂ have direct main effects, x₃ and x₄ interact nonlinearly (x₃ · x₄), while x₅ and x₆ are pure noise features. Exact Shapley and KernelSHAP cleanly identify the true signal features and assign minimal attribution to noise, satisfying exact completeness (Σ φⱼ = f(x) - E[f(x)]).'
      }
    >
      <ControlRow label="Explainer settings">
        <Select
          label="Attribution method"
          value={methodChoice}
          options={METHODS}
          onChange={(v) => setMethodChoice(v as any)}
        />
        <Select
          label="Instance to explain"
          value={String(sampleIdx)}
          options={Array.from({ length: 8 }, (_, i) => ({
            value: String(i),
            label: `Sample point #${i + 1}`,
          }))}
          onChange={(v) => setSampleIdx(Number(v))}
        />
      </ControlRow>

      <Plots>
        <Plot x={localX} y={localY} title={`Local attribution φ for sample #${sampleIdx + 1}`}>
          <Bars
            x={barIndices}
            y={attributions}
            slot={0}
            width={0.6}
          />
        </Plot>

        <Plot x={globalX} y={globalY} title="Global feature importance (mean |Shapley|)">
          <Bars
            x={barIndices}
            y={globalImportance}
            slot={1}
            width={0.6}
          />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Prediction f(x)"
          value={formatNumber(Number(prediction.toFixed(3)))}
        />
        <Readout
          label="Base value E[f(x)]"
          value={formatNumber(Number(baseValue.toFixed(3)))}
        />
        <Readout
          label="Sum of attributions Σ φⱼ"
          value={formatNumber(Number(sumAttr.toFixed(3)))}
        />
        <Readout
          label="Completeness gap"
          value={formatNumber(Number(Math.abs(baseValue + sumAttr - prediction).toFixed(4)))}
        />
      </ControlRow>
    </Figure>
  )
}
