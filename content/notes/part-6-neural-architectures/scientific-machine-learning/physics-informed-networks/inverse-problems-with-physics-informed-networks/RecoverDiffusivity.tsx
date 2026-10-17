import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { heatProblem, pinnVariantRun, type PinnVariantSnapshot } from 'aifn-methods/neural/pinn'
import { FixedHeight, pick, rowsOf, useSlicedRuns } from '../_shared/useSlicedRuns'

const PANEL = 230
const KAPPA = 0.1

type Setup = {
  observations: number
  noise: number
  window: number
  guess: number
  initialKnown: boolean
  adamSteps: number
  rate: number
  lbfgsSteps: number
  seed: number
}

/** Noisy observations of the heat equation's solution at uniform random points of [−1, 1] × [0, window]. */
function observe(s: Setup) {
  const problem = heatProblem(KAPPA)
  const r = stream(`inverse-heat/${s.seed}`)
  const x = toFlat(uniform(r, -1, 1, { shape: [s.observations] }) as Tensor)
  const t = toFlat(uniform(r, 0, s.window, { shape: [s.observations] }) as Tensor)
  const e = toFlat(normal(r, 0, s.noise, { shape: [s.observations] }) as Tensor)
  return { x, t, y: x.map((xi, i) => problem.solution!(xi, t[i]) + e[i]) }
}

/**
 * The diffusivity κ of the heat equation u_t = κu_xx recovered by a physics-informed network from noisy observations:
 * κ is a trainable parameter next to the weights, and the loss is the residual plus the misfits to the conditions and
 * to the data.
 */
export function RecoverDiffusivity() {
  const state = useFigureState({
    observations: int(50, { ge: 2, le: 1000, suggestions: [10, 50, 200], label: 'observations' }),
    noise: float(0.01, { ge: 0, le: 1, suggestions: [0, 0.01, 0.1, 0.3], label: 'noise σ' }),
    window: float(1, { ge: 0.01, le: 1, suggestions: [0.05, 0.3, 1], label: 'data up to t' }),
    guess: float(0.5, { ge: 0.001, le: 10, suggestions: [0.02, 0.5, 2], label: 'starting κ' }),
    initialKnown: toggle(true, 'initial profile known'),
    adamSteps: int(300, { ge: 0, le: 10000, suggestions: [300, 1000, 3000], label: 'Adam steps' }),
    rate: float(0.01, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.01, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(200, { ge: 0, le: 2000, suggestions: [0, 200, 300, 1000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed of the observations' }),
  })
  const { observations, noise, window, guess, initialKnown, adamSteps, rate, lbfgsSteps, seed } = state
  const setup = useMemo<Setup>(
    () => ({ observations, noise, window, guess, initialKnown, adamSteps, rate, lbfgsSteps, seed }),
    [observations, noise, window, guess, initialKnown, adamSteps, rate, lbfgsSteps, seed],
  )
  const data = useMemo(() => observe(setup), [setup])
  const { snapshots, done } = useSlicedRuns<PinnVariantSnapshot>(setup, () => [
    pinnVariantRun(heatProblem(KAPPA), {
      hidden: [12, 12],
      collocation: 64,
      trainable: { kappa: setup.guess },
      data,
      drop: setup.initialKnown ? [] : ['initial'],
      adamSteps: setup.adamSteps,
      learningRate: setup.rate,
      lbfgsSteps: setup.lbfgsSteps,
      every: 25,
      grid: [41, 21],
    }),
  ])
  const run = snapshots[0]
  const count = Math.max(1, run?.length ?? 0)
  const [pos, setPos] = useState({ setup, at: 0 })
  const at = pos.setup === setup ? Math.min(pos.at, count - 1) : 0
  const s = pick(run, at)
  // The end of the data window follows the pointer while dragged and redraws the data when let go.
  const [draft, setDraft] = useState<number | null>(null)
  const end = draft ?? window

  const view = useMemo(() => {
    if (!s || !s.solution) return null
    const nx = s.xs.length
    const last = s.prediction.length - nx
    const kappa = s.coefficients.kappa
    return {
      xs: s.xs,
      ts: s.ts,
      exact: rowsOf(s.solution, nx),
      steps: kappa.map((_, i) => i + 1),
      kappa,
      errorSteps: s.errorSteps,
      error: s.error2.map((e) => Math.max(1e-4, e)),
      finalExact: s.solution.slice(last),
      finalNetwork: s.prediction.slice(last),
    }
  }, [s])

  const total = Math.max(10, adamSteps + lbfgsSteps)
  const xA = useAxis({ label: 'x', range: [-1, 1] })
  const tA = useAxis({ label: 't', range: [0, 1] })
  const sB = useAxis({ label: 'step', range: [0, total] })
  const kB = useAxis({ label: 'κ', log: true, range: [0.01, 10] })
  const sC = useAxis({ label: 'step', range: [0, total] })
  const eC = useAxis({ label: 'relative error', log: true, range: [1e-4, 10] })
  const xD = useAxis({ label: 'x', range: [-1, 1] })
  const uD = useAxis({ label: 'u(x, 1)', range: [-0.6, 0.6] })
  const estimate = s?.coefficients.kappa.at(-1)

  return (
    <Figure
      title="Fitting a coefficient with the solution"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`The heat equation u_t = κu_xx on [−1, 1] × [0, 1] with u(x, 0) = sin πx and zero ends, true κ = ${KAPPA}. The observations are the exact solution plus Gaussian noise of standard deviation σ at uniform random points with t up to the data window. A network with two hidden layers of 12 tanh units and the coefficient κ = exp(s), with s trainable, minimise the mean squared residual at 64 collocation points plus the misfits to the conditions and to the observations, by Adam, then L-BFGS. Drag the horizontal line top left to shorten or lengthen the data window. Switch off the initial profile to make the observations carry it. Top right: the estimate of κ against the step, on a log scale, with the true value dashed. Bottom left: the network's relative L2 error against the exact solution. Bottom right: the solution and the network at t = 1, where a wrong κ shows as a wrong amplitude.`}
      controls={
        <Player
          value={at}
          onChange={(a) => setPos({ setup, at: a })}
          count={count}
          label="checkpoint"
          format={(a) => {
            const c = pick(run, a)
            return c ? `step ${c.done} (${c.phase === 'adam' ? 'Adam' : 'L-BFGS'})` : 'step 0'
          }}
        />
      }
      readouts={
        <>
          <Readout label="step" value={s ? `${s.done}${done ? '' : ', training…'}` : '…'} />
          <Readout label="estimated κ" value={estimate === undefined ? '…' : estimate.toFixed(4)} />
          <Readout
            label="relative error of κ"
            value={estimate === undefined ? '…' : `${((100 * Math.abs(estimate - KAPPA)) / KAPPA).toFixed(1)}%`}
          />
          <Readout
            label="error of the solution"
            value={s?.error2.length ? `${(100 * (s.error2.at(-1) ?? NaN)).toFixed(2)}%` : '…'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={tA} title="exact solution and observations">
            {view && <Raster x={view.xs} y={view.ts} z={view.exact} scale="diverging" range={[-1, 1]} valueLabel="u" />}
            {data.x.length > 0 && <Points name="observations" x={data.x} y={data.t} slot={2} size={5} />}
            <Handle
              kind="y"
              at={end}
              label="data up to"
              onDrag={(t) => setDraft(Math.min(1, Math.max(0.01, Math.round(t * 100) / 100)))}
              onRelease={() => {
                if (draft !== null) state.set('window', draft)
                setDraft(null)
              }}
            />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sB} y={kB} title="estimate of κ">
            <Curve name="true κ" x={[0, total]} y={[KAPPA, KAPPA]} muted dashed />
            {view && <Curve name="estimate" x={view.steps} y={view.kappa} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sC} y={eC} title="error of the solution">
            {view && <Curve name="relative error" x={view.errorSteps} y={view.error} slot={0} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xD} y={uD} title="at t = 1">
            {view && <Curve name="exact" x={view.xs} y={view.finalExact} muted dashed />}
            {view && <Curve name="network" x={view.xs} y={view.finalNetwork} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
