import { useMemo, useState } from 'react'
import { Normal } from 'aifn/probability/distributions'
import {
  clutterEp,
  clutterLogLikelihood,
  clutterPosterior,
  sampleClutter,
} from 'aifn-applied/inference/mixture-models'
import { epLogEvidence, expectationPropagation } from 'aifn/inference/expectation-propagation'
import { child, stream } from 'aifn/foundation/random'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plot,
  Area,
  Curve,
  Rug,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const GRID = linspace(-4, 8, 481)
const THETA = toFlat(GRID)
const density = (mean: number, variance: number) =>
  Number.isFinite(mean) && variance > 0 ? toFlat(Normal(mean, Math.sqrt(variance)).prob(GRID)) : THETA.map(() => NaN)

export function ClutterFigureExplorer() {
  const [n, setN] = useState(15)
  const [w, setW] = useState(0.3)
  const [damping, setDamping] = useState(0)
  const [step, setStep] = useState(0)

  const seed = 3
  const problem = useMemo(() => ({ weight: w }), [w])

  const x = useMemo(
    () => Array.from(sampleClutter(child(stream('content/clutter'), seed), n, 2, problem).data),
    [n, seed, problem],
  )

  const run = useMemo(
    () => trace(expectationPropagation(clutterEp(x, problem, { damping })), undefined, 30 * x.length),
    [x, problem, damping],
  )

  const exact = useMemo(() => clutterPosterior(x, problem, { lower: -4, upper: 8, points: 481 }), [x, problem])

  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const q = s.posterior
  const i = s.site

  const curves = useMemo(() => {
    const factor = i >= 0 ? THETA.map((t) => Math.exp(clutterLogLikelihood(t, [x[i]], problem))) : []
    const qd = density(q.mean, q.variance)
    const top = Math.max(...qd)
    const fmax = Math.max(...factor, 1e-300)
    return {
      qd,
      factor: factor.map((f) => (f / fmax) * top),
      cavity: i >= 0 ? density(s.cavity.mean, s.cavity.variance) : null,
      tilted: i >= 0 ? density(s.tiltedMoments.mean, s.tiltedMoments.variance) : null,
    }
  }, [s, q, i, x, problem])

  const exactDensity = useMemo(() => toFlat(exact.density), [exact])
  const theta = useAxis({ label: 'parameter θ', range: [-4, 8] })
  const dens = useAxis({ label: 'posterior density', hold: 'union', key: `${n}-${seed}-${w}` })

  const formatStep = (p: number) => {
    const k = run.index[p]
    return k === 0 ? 'prior' : `sweep ${Math.floor((k - 1) / x.length) + 1}, site ${((k - 1) % x.length) + 1}`
  }

  return (
    <Figure
      title="Expectation propagation on Minka's clutter problem"
      purpose="Each site update divides one factor's Gaussian site out of q to form the cavity q∖i, multiplies the exact non-Gaussian observation factor back in, and moment-matches the tilted distribution back to a Gaussian."
      defaultSize="L"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Sample size n"
            value={n}
            onChange={(val) => {
              setN(val)
              setStep(0)
            }}
            min={5}
            max={35}
            step={5}
            suggestions={[10, 15, 25]}
          />
          <NumberSelector
            label="Clutter prob w"
            value={w}
            onChange={(val) => {
              setW(val)
              setStep(0)
            }}
            min={0.05}
            max={0.7}
            step={0.05}
            suggestions={[0.1, 0.3, 0.5]}
          />
          <NumberSelector
            label="Damping"
            value={damping}
            onChange={setDamping}
            min={0}
            max={0.8}
            step={0.1}
            suggestions={[0, 0.2, 0.5]}
          />
          <Player
            label="Site update step"
            value={Math.min(step, run.steps.length - 1)}
            onChange={setStep}
            count={run.steps.length}
            format={formatStep}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="Current site x_i" value={i >= 0 ? formatNumber(x[i]) : '—'} />
          <Readout
            label="EP posterior μ ± σ"
            value={`${formatNumber(q.mean)} ± ${formatNumber(Math.sqrt(q.variance))}`}
          />
          <Readout
            label="Exact posterior μ ± σ"
            value={`${formatNumber(exact.mean)} ± ${formatNumber(Math.sqrt(exact.variance))}`}
          />
          <Readout
            label="Log evidence (EP / exact)"
            value={`${formatNumber(epLogEvidence(s))} / ${formatNumber(exact.logEvidence)}`}
          />
        </>
      }
      caption="Shaded area: exact multimodal posterior over θ. Curves show the EP Gaussian posterior (blue solid), the cavity distribution with site i excluded (slot 2), the scaled likelihood factor for observation x_i (dashed), and the moment-matched tilted distribution (slot 1). Ticks along the axis mark sampled data points."
    >
      <Plot x={theta} y={dens}>
        <Area name="exact posterior" x={THETA} y={exactDensity} opacity={0.15} />
        {i >= 0 && <Curve name={`factor t_${i}(θ)`} x={THETA} y={curves.factor} slot={3} dashed />}
        {curves.cavity && <Curve name="cavity q∖i" x={THETA} y={curves.cavity} slot={2} />}
        {curves.tilted && <Curve name="tilted moment-matched" x={THETA} y={curves.tilted} slot={1} dashed />}
        <Curve name="EP posterior q" x={THETA} y={curves.qd} slot={0} />
        <Rug name="data" values={x} emphasis length={12} />
      </Plot>
    </Figure>
  )
}
