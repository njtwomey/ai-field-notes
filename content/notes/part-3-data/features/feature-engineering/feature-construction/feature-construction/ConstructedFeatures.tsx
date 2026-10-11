import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
  choice,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { interactionTask } from 'aifn-methods/data/synthetic'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { linearRegression } from 'aifn-methods/learning/linear'
import { pipeline } from 'aifn-compute/learning/compose'
import { crossValidate, kFold, type Fittable } from 'aifn-compute/learning/validate'
import type { Decides } from 'aifn-compute/learning/estimators'
import { r2Score } from 'aifn-compute/learning/metrics'
import { stream } from 'aifn-compute/foundation/random'
import {
  mul,
  sin,
  slice,
  square,
  stack,
  sub,
  toFlat,
  unwrap,
  variance,
  type Tensor,
} from 'aifn-compute/foundation/tensor'

const PANEL = 280

const SETS = [
  { value: 'raw', label: 'raw: x₁, x₂, x₃' },
  { value: 'products', label: 'raw + pairwise products' },
  { value: 'degree2', label: 'all monomials of degree ≤ 2' },
  { value: 'degree3', label: 'all monomials of degree ≤ 3' },
  { value: 'constructed', label: 'constructed: sin πx₁, x₂², x₃, x₁x₂' },
] as const
type SetKey = (typeof SETS)[number]['value']
const SHORT = ['raw', '+ products', 'degree ≤ 2', 'degree ≤ 3', 'constructed']

const AGAINST = [
  { value: 'x1', label: 'x₁' },
  { value: 'x2', label: 'x₂' },
  { value: 'x1x2', label: 'x₁x₂' },
] as const

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const column = (x: Tensor, j: number) => slice(x, null, j) as Tensor

/** The inputs a feature set gives the linear model, and the model (with any expansion) that fits on them. */
type Regressor = Fittable<Tensor, Decides<Tensor, Tensor>>

function featureSet(key: SetKey, x: Tensor): { x: Tensor; model: Regressor; width: number } {
  const ols = linearRegression()
  if (key === 'raw') return { x, model: ols, width: 3 }
  if (key === 'products')
    return { x, model: pipeline(polynomialFeatures({ degree: 2, interactionOnly: true }), ols), width: 6 }
  if (key === 'degree2') return { x, model: pipeline(polynomialFeatures({ degree: 2 }), ols), width: 9 }
  if (key === 'degree3') return { x, model: pipeline(polynomialFeatures({ degree: 3 }), ols), width: 19 }
  const [x1, x2, x3] = [0, 1, 2].map((j) => column(x, j))
  const built = stack([sin(mul(x1, Math.PI)) as Tensor, square(x2) as Tensor, x3, mul(x1, x2) as Tensor], 1)
  return { x: built, model: ols, width: 4 }
}

/** Linear regression on five feature sets built from the same three inputs, scored by cross-validated R². */
export function ConstructedFeatures() {
  const state = useFigureState({
    features: choice(SETS, 'raw', { label: 'feature set' }),
    against: choice(AGAINST, 'x1', { label: 'residuals against' }),
    interaction: slider(-2, 2, 1.5, { step: 0.1, label: 'interaction strength γ' }),
    noise: slider(0, 1, 0.1, { step: 0.05, label: 'noise sd' }),
    n: int(300, { ge: 40, le: 3000, suggestions: [100, 300, 1000], label: 'rows' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { features, against, interaction, noise, n, seed } = state
  const data = useMemo(
    () => interactionTask(stream(`feature-construction/${seed}`), { n, interaction, noise }),
    [n, interaction, noise, seed],
  )
  const scores = useMemo(
    () =>
      SETS.map(({ value }) => {
        const { x, model, width } = featureSet(value, data.x)
        const cv = crossValidate(model, { kind: 'dataset', x, y: data.y! }, kFold({ k: 5, shuffle: true }), [r2Score], {
          stream: stream(`feature-construction/cv/${seed}`),
        })
        return { r2: cv.mean.r2Score, width, oof: cv.outOfFold.decide as Tensor }
      }),
    [data, seed],
  )
  // The best R² any model can reach: the share of the target's variance that is signal, Var f / (Var f + noise²).
  const ceiling = useMemo(() => {
    const signal = Number(unwrap(variance(data.f!)))
    return signal / (signal + noise * noise)
  }, [data, noise])
  const chosen = SETS.findIndex((s) => s.value === features)
  const residual = useMemo(() => {
    const r = Array.from(toFlat(sub(data.y!, scores[chosen].oof) as Tensor))
    const [x1, x2] = [column(data.x, 0), column(data.x, 1)]
    const axis = against === 'x1' ? x1 : against === 'x2' ? x2 : (mul(x1, x2) as Tensor)
    return { x: Array.from(toFlat(axis)), r }
  }, [data, scores, chosen, against])

  const setAxis = useAxis({ label: 'feature set', categories: SHORT })
  const r2Axis = useAxis({ label: 'cross-validated R²', range: [0, 1] })
  const rx = useAxis({ label: AGAINST.find((a) => a.value === against)!.label })
  const ry = useAxis({ label: 'out-of-fold residual', key: features, hold: 'initial' })
  const positions = SETS.map((_, i) => i)
  return (
    <Figure
      title="Constructing the features a linear model needs"
      state={state}
      hoverReadout={false}
      caption="The target is y = sin(πx₁) + x₂² + ½x₃ + γx₁x₂ + noise, with three independent inputs uniform on [−1, 1]. Each bar is the 5-fold cross-validated R² of ordinary least squares on one feature set built from the same three inputs. The constructed set has four features chosen from the target's structure; the degree-3 expansion reaches a similar score with nineteen. Right: the out-of-fold residuals of the chosen set against an input. A trend in the residuals is structure the features have not captured: a sine-shaped curve against x₁ and a slope against x₁x₂ for the raw features."
      readouts={
        <>
          <Readout label="features" value={String(scores[chosen].width)} />
          <Readout label="cross-validated R²" value={formatNumber(scores[chosen].r2)} />
          <Readout label="noise ceiling of R²" value={formatNumber(ceiling)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={setAxis} y={r2Axis} title="Cross-validated R² by feature set">
            <Bars
              name="other sets"
              x={positions.filter((i) => i !== chosen)}
              y={scores.filter((_, i) => i !== chosen).map((s) => Math.max(0, s.r2))}
              width={0.6}
              muted
            />
            <Bars name="chosen set" x={[chosen]} y={[Math.max(0, scores[chosen].r2)]} width={0.6} slot={0} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={rx} y={ry} title="Residuals of the chosen set">
            <Points name="residual" x={residual.x} y={residual.r} slot={0} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
