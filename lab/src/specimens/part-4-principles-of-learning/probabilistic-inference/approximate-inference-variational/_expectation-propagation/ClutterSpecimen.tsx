import { Normal } from 'aifn-compute/probability/distributions'
import { clutterEp, clutterLogLikelihood, clutterPosterior, sampleClutter } from 'aifn-methods/inference/mixture-models'
import { epLogEvidence, expectationPropagation } from 'aifn-compute/inference/expectation-propagation'
import { child, stream } from 'aifn-compute/foundation/random'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { row, slider, toggle, useFigureState } from 'aifn-render/state'
import { Area, Curve, Plot, Readout, Rug, formatNumber, useAxis } from 'aifn-render/viz'

const GRID = linspace(-4, 8, 481)
const THETA = toFlat(GRID)
const density = (mean: number, variance: number) =>
  Number.isFinite(mean) && variance > 0 ? toFlat(Normal(mean, Math.sqrt(variance)).prob(GRID)) : THETA.map(() => NaN)

/** EP on Minka's clutter problem, one site update per step: the cavity, the tilted distribution and the new q. */
export function ClutterSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      n: slider(3, 40, 15, { label: 'points n', step: 1 }),
      w: slider(0.05, 0.8, 0.3, { label: 'clutter probability w', step: 0.05 }),
      seed: slider(1, 30, 3, { label: 'seed', step: 1 }),
    }),
    ep: row('2 · EP', { damping: slider(0, 0.9, 0, { label: 'damping', step: 0.05 }) }),
    reveal: row('3 · reveal', { showExact: toggle(true, 'exact posterior') }),
  })
  const { n, w, seed } = state.data
  const { damping } = state.ep
  const { showExact } = state.reveal
  const problem = useMemo(() => ({ weight: w }), [w])
  const x = useMemo(
    () => Array.from(sampleClutter(child(stream('clutter'), seed), n, 2, problem).data),
    [n, seed, problem],
  )
  const run = useMemo(
    () => trace(expectationPropagation(clutterEp(x, problem, { damping })), undefined, 30 * x.length),
    [x, problem, damping],
  )
  const exact = useMemo(() => clutterPosterior(x, problem, { lower: -4, upper: 8, points: 481 }), [x, problem])
  const [step, setStep] = useState(0)
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
  const theta = useAxis({ label: 'θ', range: [-4, 8] })
  const dens = useAxis({ label: 'density', hold: 'union', key: `${n}-${seed}-${w}` })
  const format = (p: number) => {
    const k = run.index[p]
    return k === 0 ? 'start' : `s${Math.floor((k - 1) / x.length) + 1} i${(k - 1) % x.length}`
  }
  return (
    <Figure
      title="EP on the clutter problem, site by site"
      purpose="Each update divides a site out of q, multiplies the exact factor back in, and projects the tilted result onto a Gaussian."
      defaultSize="L"
      state={state}
      controls={
        <div className="col-span-full">
          <Player
            label="4 · update (sweep s, site i)"
            value={Math.min(step, run.steps.length - 1)}
            onChange={setStep}
            count={run.steps.length}
            format={format}
          />
        </div>
      }
      readouts={{
        'this update': (
          <>
            <Readout label="x_i" value={i >= 0 ? formatNumber(x[i]) : '—'} />
            <Readout label="EP mean / sd" value={`${formatNumber(q.mean)} / ${formatNumber(Math.sqrt(q.variance))}`} />
            <Readout
              label="exact mean / sd"
              value={`${formatNumber(exact.mean)} / ${formatNumber(Math.sqrt(exact.variance))}`}
            />
          </>
        ),
        convergence: (
          <>
            <Readout
              label="last sweep's largest change"
              value={Number.isFinite(s.lastSweepChange) ? formatNumber(s.lastSweepChange) : '—'}
            />
            <Readout
              label="log evidence EP / exact"
              value={`${formatNumber(epLogEvidence(s))} / ${formatNumber(exact.logEvidence)}`}
            />
            <Readout label="skipped updates" value={s.totalSkipped} />
          </>
        ),
      }}
      caption="θ ~ N(0, 100); each point is N(θ, 1) with probability 1 − w and clutter N(0, 10) with probability w (Minka 2001). The data are the ink ticks along the axis. Play from the prior. At each update the cavity is q with site i removed, the dashed factor is the exact clutter likelihood of x_i, and the tilted distribution (cavity × factor) is replaced by the Gaussian with its mean and variance. The first sweep is assumed density filtering; later sweeps revisit each site until nothing moves. The evidence estimate uses every site's normaliser, so it is meaningful only after the first sweep."
    >
      <Plot x={theta} y={dens}>
        {showExact && <Area name="exact posterior" x={THETA} y={exactDensity} muted />}
        {i >= 0 && <Curve name={`factor t_${i}(θ), scaled`} x={THETA} y={curves.factor} slot={3} dashed />}
        {curves.cavity && <Curve name="cavity q∖i" x={THETA} y={curves.cavity} slot={2} />}
        {curves.tilted && <Curve name="tilted, moment-matched" x={THETA} y={curves.tilted} slot={1} dashed />}
        <Curve name="EP posterior q" x={THETA} y={curves.qd} slot={0} />
        <Rug name="data" values={x} emphasis length={12} />
      </Plot>
    </Figure>
  )
}
