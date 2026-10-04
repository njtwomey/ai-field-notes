import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Phi, gaussPdf, grid } from '../_shared/gaussian'
import { predict, prior, update, type Belief } from '../_shared/probit'
import { stream as randomStream, uniform } from 'aifn/foundation/random'

/** Features: 0 bias, 1–3 the ad (A, B, C), 4–5 the display position (top, side). */
const ADS = ['ad A', 'ad B', 'ad C']
const TRUE_W = [-1, 0.4, 0, -0.4, 0.3, -0.3]
const AD_FREQ = [0.5, 0.4, 0.1]
const BETA = 1
const N = 800
const PRIOR_VAR = 0.5

type Impression = { active: number[]; click: boolean }

/** A seeded stream of impressions and the belief after each one. */
function simulate() {
  const r = randomStream(3)
  const stream: Impression[] = []
  const beliefs: Belief[] = [prior(TRUE_W.length, 0, PRIOR_VAR)]
  for (let n = 0; n < N; n++) {
    const u = uniform(r)
    const ad = u < AD_FREQ[0] ? 1 : u < AD_FREQ[0] + AD_FREQ[1] ? 2 : 3
    const pos = uniform(r) < 0.5 ? 4 : 5
    const active = [0, ad, pos]
    const ctr = Phi(active.reduce((s, i) => s + TRUE_W[i], 0) / BETA)
    const click = uniform(r) < ctr
    stream.push({ active, click })
    beliefs.push(update(beliefs[n], active, click ? 1 : -1, BETA))
  }
  return { stream, beliefs }
}

const { stream: STREAM, beliefs: BELIEFS } = simulate()
const W = grid(-2.5, 2.5, 201)
const STEPS = Array.from({ length: N + 1 }, (_, i) => i)
const topCtr = (b: Belief, ad: number) => predict(b, [0, ad, 4], BETA)
const TRUE_CTR = [1, 2, 3].map((ad) => Phi((TRUE_W[0] + TRUE_W[ad] + TRUE_W[4]) / BETA))
const CTR_PATHS = [1, 2, 3].map((ad) => BELIEFS.map((b) => topCtr(b, ad)))

/** Gaussian weight beliefs narrowing, and predicted click-through rates settling, as impressions arrive. */
export function AdPredictorStream() {
  const state = useFigureState({
    step: int(40, { min: 0, max: N, step: 1, label: 'impressions seen', format: (v) => String(v) }),
  })
  const b = BELIEFS[state.step]

  const densities = useMemo((): SeriesSpec[] => {
    const out: SeriesSpec[] = []
    ;[1, 2, 3].forEach((ad, k) => {
      out.push({ name: ADS[k], type: 'line', x: W, y: W.map((w) => gaussPdf(w, b.mean[ad], b.variance[ad])), slot: k })
    })
    return out
  }, [b])

  const paths = useMemo((): SeriesSpec[] => {
    const out: SeriesSpec[] = CTR_PATHS.map((y, k) => ({ name: ADS[k], type: 'line', x: STEPS, y, slot: k }))
    TRUE_CTR.forEach((c, k) =>
      out.push({ name: `${ADS[k]} true`, type: 'line', x: [0, N], y: [c, c], slot: k, dashed: true }),
    )
    return out
  }, [])

  const seen = [1, 2, 3].map((ad) => STREAM.slice(0, state.step).filter((s) => s.active[1] === ad).length)
  const clicks = STREAM.slice(0, state.step).filter((s) => s.click).length

  const xAxis = useAxis({ label: 'ad weight', range: [-2.5, 2.5] })
  const yAxis = useAxis({ label: 'belief density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'impressions', range: [0, N] })
  const yAxis2 = useAxis({ label: 'predicted CTR, top position', range: [0, 1] })
  return (
    <Figure
      title="Online probit regression on a stream of ad impressions"
      state={state}
      caption="Each impression activates a bias weight, one of three ads and one of two positions; its click is drawn from the true model. After every impression the active weights move by the update in this note. Left: the Gaussian beliefs about the three ad weights. Right: the predicted click-through rate of each ad in the top position, with the true rates dashed. Ad C is shown one time in ten, so its belief narrows slowest. The weights themselves need not approach the generating values, because adding a constant to every ad weight and subtracting it from the bias changes no prediction; the prior settles that choice. Step with the arrows or drag the line on the right."

      readouts={
        <>
          {ADS.map((name, k) => (
            <Readout
              key={name}
              label={`${name}: shown ${seen[k]}, σ`}
              value={formatNumber(Math.sqrt(b.variance[k + 1]))}
            />
          ))}
          <Readout label="clicks so far" value={clicks} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          {seriesLayers(densities)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          {seriesLayers(paths)}
          <Handle {...state.handle('step', { label: 'now' })} />
        </Plot>
      </div>
    </Figure>
  )
}
