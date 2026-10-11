import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Density,
  Figure,
  formatNumber,
  FrameContext,
  Histogram,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { exp, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'
import { skewness } from 'aifn-compute/probability/stats'
import { boxCox, fitTransform, powerTransform, standardScaler, yeoJohnson } from 'aifn-methods/learning/preprocessing'

const N = 400
const PANEL = 240
const Z = toFlat(normals(stream('power-transforms/data'), [N]))
const STANDARD_NORMAL = Normal(0, 1)

const METHODS = [
  { value: 'box-cox', label: 'Box–Cox' },
  { value: 'yeo-johnson', label: 'Yeo–Johnson' },
] as const

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const column = (v: ArrayLike<number>) => fromData(Float64Array.from(v), [v.length, 1])

/** A skewed sample before and after a power transform, with λ by maximum likelihood or by hand. */
export function PowerTransformFit() {
  const state = useFigureState({
    sigma: slider(0.1, 1.5, 0.8, { step: 0.05, label: 'skew σ' }),
    shift: slider(-3, 3, 0, { step: 0.1, label: 'shift' }),
    method: choice(METHODS, 'box-cox', { label: 'transform' }),
    fitted: setting(true, 'λ by maximum likelihood'),
    lambda: slider(-1, 2, 1, { step: 0.05, label: 'λ by hand' }),
  })
  const { sigma, shift, method, fitted, lambda } = state
  const view = useMemo(() => {
    // Log-normal draws exp(0.5 + σZ), moved by `shift`: σ sets the skew, a negative shift makes some values negative.
    const x = Array.from(
      toFlat(
        exp(
          fromData(
            Float64Array.from(Z, (z) => 0.5 + sigma * z),
            [N],
          ),
        ),
      ),
    ).map((v) => v + shift)
    const positive = Math.min(...x) > 0
    if (method === 'box-cox' && !positive) return { x, z: [] as number[], positive, mle: NaN, used: NaN }
    const model = powerTransform({ method }).fit({ x: column(x) })
    const mle = model.lambdas[0]
    const used = fitted ? mle : lambda
    // Either the fitted transform (which standardises), or the chosen λ followed by standardisation.
    const raw = fitted
      ? model.transform(column(x))
      : (method === 'box-cox' ? boxCox : yeoJohnson)(column(x) as Tensor, used)
    const z = fitted ? Array.from(toFlat(raw)) : Array.from(toFlat(fitTransform(standardScaler(), { x: raw }).z))
    return { x, z, positive, mle, used }
  }, [sigma, shift, method, fitted, lambda])
  const xa = useAxis({ label: 'x' })
  const xd = useAxis({ label: 'density' })
  const za = useAxis({ label: 'transformed and standardised', range: [-4, 4] })
  const zd = useAxis({ label: 'density', range: [0, 0.7] })
  const blocked = method === 'box-cox' && !view.positive
  return (
    <Figure
      title="A power transform removes skew"
      state={state}
      hoverReadout={false}
      caption="Left: 400 draws of exp(0.5 + σZ) + shift, with Z standard normal; σ sets the skew. Right: the same values after the power transform, standardised, against a standard normal density. With λ by maximum likelihood, Box–Cox picks λ close to 0, the logarithm, because log-normal data are normal after a log. Set λ by hand to see 1 leave the skew in place and values below the fitted λ over-correct it into a left skew. A negative shift makes some values negative, which Box–Cox cannot take; Yeo–Johnson still applies."
      readouts={
        <>
          <Readout label="λ by maximum likelihood" value={blocked ? '–' : formatNumber(view.mle)} />
          <Readout label="λ used" value={blocked ? '–' : formatNumber(view.used)} />
          <Readout label="skewness before" value={formatNumber(skewness(view.x))} />
          <Readout
            label="skewness after"
            value={blocked ? 'Box–Cox needs positive values' : formatNumber(skewness(view.z))}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xa} y={xd} title="data">
            <Histogram name="x" values={view.x} bins={30} slot={0} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={za} y={zd} title="after the transform">
            {view.z.length > 0 && <Histogram name="transformed" values={view.z} bins={30} slot={1} />}
            <Density name="standard normal" dist={STANDARD_NORMAL} range={[-4, 4]} dashed muted />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
