import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { add, mul, type Tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { type MlstmForget, mlstmParallel, mlstmRecurrent } from 'aifn-compute/nn/sequence'

const D = 4
/** The largest finite values of float32 and float16, as natural logs: ln(3.4·10³⁸) and ln(65504). */
const LN_MAX = { float32: Math.log(3.4028235e38), float16: Math.log(65504) }

/** The first step at which the largest unstabilised weight e^{m_t} exceeds a format's range, or null. */
const overflowAt = (m: number[], lnMax: number) => {
  const t = m.findIndex((x) => x > lnMax)
  return t < 0 ? null : t + 1
}

/** mLSTM with exponential gates: the stabiliser m_t, and the recurrent and parallel outputs, on random tokens. */
export function MlstmStabiliser() {
  const state = useFigureState({
    forget: choice<MlstmForget>(
      [
        { value: 'sigmoid', label: 'sigmoid forget gate' },
        { value: 'exp', label: 'exponential forget gate' },
      ],
      'sigmoid',
      { label: 'forget gate' },
    ),
    iBias: slider(-5, 100, 8, { step: 1, label: 'input-gate pre-activation ĩ (mean)' }),
    fBias: slider(-3, 6, 3, { step: 0.25, label: 'forget-gate pre-activation f̃ (mean)' }),
    T: int(128, { min: 8, max: 512, suggestions: [32, 128, 256, 512], label: 'length T' }),
  })
  const { forget, iBias, fBias, T } = state

  const r = useMemo(() => {
    const s = stream(2)
    const q = normals(s, [T, D])
    const k = mul(normals(s, [T, D]), 1 / Math.sqrt(D))
    const v = normals(s, [T, D])
    const iPre = add(normals(s, [T]), iBias)
    const fPre = add(mul(normals(s, [T]), 0.5), fBias)
    const rec = mlstmRecurrent(q, k, v, iPre, fPre, { forget })
    const par = mlstmParallel(q, k, v, iPre, fPre, { forget })
    const m = toFlat(rec.stabiliser as Tensor)
    const hr = toFlat(rec.outputs as Tensor)
    const hp = toFlat(par.outputs as Tensor)
    return {
      log10m: m.map((x) => x / Math.LN10),
      m,
      rec: Array.from({ length: T }, (_, t) => hr[t * D]),
      par: Array.from({ length: T }, (_, t) => hp[t * D]),
      gap: Math.max(...hr.map((x, i) => Math.abs(x - hp[i]))),
      finite: hr.every(Number.isFinite),
    }
  }, [forget, iBias, fBias, T])

  const t = useMemo(() => Array.from({ length: T }, (_, i) => i + 1), [T])
  const xAxis = useAxis({ label: 'token t', integer: true })
  const yTop = useAxis({ label: 'log₁₀ largest weight e^mₜ' })
  const yBottom = useAxis({ label: 'output h̃ₜ, channel 1' })
  const f32 = overflowAt(r.m, LN_MAX.float32)
  const f16 = overflowAt(r.m, LN_MAX.float16)

  return (
    <Figure
      title="The mLSTM stabiliser"
      state={state}
      caption="Top: the stabiliser mₜ, as log₁₀ of the largest weight iₛ fₛ₊₁⋯fₜ in the memory's sums. Without the stabiliser the memory Cₜ and normaliser nₜ hold numbers of this size; the dashed lines are the largest values float16 and float32 can hold. A large input-gate pre-activation, or an exponential forget gate with a positive pre-activation, pushes the weights past both. The stabilised memory divides every weight by e^mₜ, so its largest weight is 1 at every step. Bottom: the first output channel from the stabilised recurrence (line) and from the parallel, attention-like form (points). The two agree to rounding however large the weights grow."
      readouts={
        <>
          <Readout label="float16 overflows from" value={f16 === null ? 'never' : `t = ${f16}`} />
          <Readout label="float32 overflows from" value={f32 === null ? 'never' : `t = ${f32}`} />
          <Readout label="largest weight at the end" value={`10^${formatNumber(r.log10m[T - 1])}`} />
          <Readout label="largest |recurrent − parallel|" value={r.finite ? formatNumber(r.gap) : 'not finite'} />
        </>
      }
    >
      <Plots rows={2}>
        <Plot x={xAxis} y={yTop}>
          <Curve name="log₁₀ e^mₜ" x={t} y={r.log10m} slot={0} />
          <Annotation y={LN_MAX.float32 / Math.LN10} text="float32 limit" dashed />
          <Annotation y={LN_MAX.float16 / Math.LN10} text="float16 limit" dashed />
        </Plot>
        <Plot x={xAxis} y={yBottom}>
          <Curve name="stabilised recurrence" x={t} y={r.rec} slot={0} />
          <Points name="parallel form" x={t} y={r.par} slot={1} size={4} />
        </Plot>
      </Plots>
    </Figure>
  )
}
