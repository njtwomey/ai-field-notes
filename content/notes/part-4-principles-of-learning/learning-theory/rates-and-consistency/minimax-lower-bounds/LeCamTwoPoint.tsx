import { useMemo } from 'react'
import { arange, toFlat } from 'aifn-compute/foundation/tensor'
import { Binomial } from 'aifn-compute/probability/distributions'
import { totalVariation } from 'aifn-compute/probability/information'
import { bernoulliKl, tailBounds } from 'aifn-methods/theory/concentration'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const M_MAX = 2000
/** Sample sizes for the right panel: about 60, spread evenly on a log scale. */
const SIZES = Array.from(
  new Set(Array.from({ length: 61 }, (_, i) => Math.max(1, Math.round(Math.exp((i / 60) * Math.log(M_MAX)))))),
)
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The two count distributions of m flips of a coin with bias ½ − δ or ½ + δ, and their total variation distance. */
function counts(m: number, delta: number) {
  const k = arange(0, m + 1)
  const p0 = toFlat(Binomial(m, 0.5 - delta).prob(k))
  const p1 = toFlat(Binomial(m, 0.5 + delta).prob(k))
  return { k: toFlat(k), p0, p1, tv: Number(totalVariation(p0, p1)) }
}

/**
 * Le Cam's two-point method on a coin: no estimator of the bias can tell ½ − δ from ½ + δ with error below
 * (1 − TV)/2, where TV is the total variation distance between the two distributions of the sample. The figure draws
 * the two count distributions, whose overlap is 1 − TV, and the lower bound against m beside Hoeffding's upper bound
 * for the sample mean.
 */
export function LeCamTwoPoint() {
  const state = useFigureState({
    delta: slider(0.01, 0.2, 0.05, { step: 0.01, label: 'half-gap δ' }),
    m: int(100, { ge: 1, le: M_MAX, suggestions: [10, 50, 100, 400], label: 'flips m' }),
  })
  const { delta, m } = state

  const curve = useMemo(() => {
    const lower = SIZES.map((n) => (1 - counts(n, delta).tv) / 2)
    const upper = SIZES.map((n) => tailBounds([delta], n, 0.5, 0.25).hoeffding[0])
    return { lower, upper }
  }, [delta])
  const now = useMemo(() => {
    const c = counts(m, delta)
    const overlap = Array.from(c.p0, (v, i) => Math.min(v, c.p1[i]))
    return { ...c, overlap }
  }, [m, delta])
  const kl = m * bernoulliKl(0.5 - delta, 0.5 + delta)
  const lowerNow = (1 - now.tv) / 2
  const upperNow = tailBounds([delta], m, 0.5, 0.25).hoeffding[0]

  const kAxis = useAxis({ label: 'number of heads K', range: [0, m], key: String(m) })
  const pAxis = useAxis({ label: 'probability', hold: 'union', key: `${m}/${delta}` })
  const mAxis = useAxis({ label: 'flips m', range: [1, M_MAX], log: true, format: (v) => String(Math.round(v)) })
  const eAxis = useAxis({ label: 'probability of missing p by δ or more', range: [0, 1] })
  return (
    <Figure
      title="Two coins that m flips cannot tell apart"
      purpose="Le Cam's two-point method: if the samples from two hypotheses overlap, every estimator errs on one of them, with probability at least (1 − TV)/2."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="total variation TV" value={fmt(now.tv)} />
          <Readout label="KL = m · kl(½ − δ ‖ ½ + δ)" value={fmt(kl)} />
          <Readout label="lower bound (1 − TV)/2" value={fmt(lowerNow)} />
          <Readout label="sample mean, Hoeffding bound" value={fmt(Math.min(1, upperNow))} />
        </>
      }
      caption="A coin has bias p = ½ − δ or p = ½ + δ, and an estimator sees m flips. Left: the distributions of the number of heads under the two hypotheses; the shaded overlap has total mass 1 − TV. Right: the ink curve is Le Cam's lower bound (1 − TV)/2 on the probability that the best estimator misses p by δ or more, on whichever hypothesis is worse for it; the coloured curve is Hoeffding's bound 2 exp(−2mδ²) on that probability for the sample mean, an upper bound on what is achievable. Drag the vertical line to change m. At δ = 0.05 the lower bound falls to 0.05 only at m = 269 flips, while Hoeffding's guarantee for the sample mean needs 738; the true minimax sample size lies between."
    >
      <Plots cols={2}>
        <Plot x={kAxis} y={pAxis} title="the two samples">
          <Area name="overlap 1 − TV" x={now.k} y={now.overlap} muted />
          <Curve name="p = ½ − δ" x={now.k} y={now.p0} slot={0} showPoints={m <= 60} />
          <Curve name="p = ½ + δ" x={now.k} y={now.p1} slot={1} showPoints={m <= 60} />
        </Plot>
        <Plot x={mAxis} y={eAxis} title="error against m">
          <Curve name="lower bound (1 − TV)/2" x={SIZES} y={curve.lower} emphasis />
          <Curve name="sample mean (Hoeffding)" x={SIZES} y={curve.upper} slot={2} />
          <Handle
            kind="x"
            at={m}
            onDrag={(v) => state.set('m', Math.max(1, Math.min(M_MAX, Math.round(v))))}
            label="m"
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
