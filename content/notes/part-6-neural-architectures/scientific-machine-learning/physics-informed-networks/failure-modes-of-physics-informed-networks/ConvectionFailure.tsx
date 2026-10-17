import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  int,
  Player,
  Plot,
  Raster,
  Readout,
  slider,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  convectionProblem,
  pinnVariantRun,
  type PinnVariantOptions,
  type PinnVariantSnapshot,
} from 'aifn-methods/neural/pinn'
import { FixedHeight, pick, rowsOf, useSlicedRuns } from '../_shared/useSlicedRuns'

const PANEL = 230
const TWO_PI = 2 * Math.PI

type Setup = {
  beta: number
  fourier: boolean
  causal: boolean
  curriculum: boolean
  wide: boolean
  collocation: number
  adamSteps: number
  rate: number
  lbfgsSteps: number
  seed: number
}

/**
 * The convection equation u_t + βu_x = 0 from u(x, 0) = sin x, solved by a physics-informed network as the speed β
 * grows, with Fourier-feature inputs, causal weights in time and a curriculum in β as switchable remedies.
 */
export function ConvectionFailure() {
  const state = useFigureState({
    beta: slider(1, 50, 20, { step: 1, label: 'speed β' }),
    fourier: toggle(false, 'Fourier-feature inputs'),
    causal: toggle(false, 'causal weights in time'),
    curriculum: toggle(false, 'curriculum in β'),
    wide: toggle(false, 'wide network (32 units)'),
    collocation: int(128, { ge: 32, le: 2048, suggestions: [128, 256, 512], label: 'collocation points' }),
    adamSteps: int(300, { ge: 0, le: 10000, suggestions: [300, 1000, 3000], label: 'Adam steps' }),
    rate: float(0.005, { ge: 1e-4, le: 0.1, suggestions: [0.001, 0.005, 0.02], label: 'Adam learning rate' }),
    lbfgsSteps: int(100, { ge: 0, le: 2000, suggestions: [0, 100, 300, 1000], label: 'L-BFGS steps' }),
    seed: int(0, { ge: 0, le: 999, label: 'seed' }),
  })
  const { beta, fourier, causal, curriculum, wide, collocation, adamSteps, rate, lbfgsSteps, seed } = state
  const setup = useMemo<Setup>(
    () => ({ beta, fourier, causal, curriculum, wide, collocation, adamSteps, rate, lbfgsSteps, seed }),
    [beta, fourier, causal, curriculum, wide, collocation, adamSteps, rate, lbfgsSteps, seed],
  )
  const { snapshots, done } = useSlicedRuns<PinnVariantSnapshot>(setup, () => {
    const options: PinnVariantOptions = {
      hidden: setup.wide ? [32, 32] : [16, 16],
      collocation: setup.collocation,
      fourier: setup.fourier ? { count: 16, scale: 2 } : null,
      causal: setup.causal ? { epsilon: 10, bins: 16 } : null,
      curriculum: setup.curriculum ? { name: 'beta', from: 1, steps: Math.round(0.7 * setup.adamSteps) } : null,
      adamSteps: setup.adamSteps,
      learningRate: setup.rate,
      lbfgsSteps: setup.lbfgsSteps,
      seed: setup.seed,
      every: 25,
      grid: [41, 21],
    }
    return [pinnVariantRun(convectionProblem(setup.beta), options)]
  })
  const run = snapshots[0]
  const count = Math.max(1, run?.length ?? 0)
  const [pos, setPos] = useState({ setup, at: 0 })
  const at = pos.setup === setup ? Math.min(pos.at, count - 1) : 0
  const s = pick(run, at)

  const view = useMemo(() => {
    if (!s || !s.solution) return null
    const nx = s.xs.length
    const byTime = rowsOf(s.residual, nx).map((row) => Math.max(1e-8, row.reduce((m, r) => m + r * r, 0) / nx))
    return {
      xs: s.xs,
      ts: s.ts,
      exact: rowsOf(s.solution, nx),
      prediction: rowsOf(s.prediction, nx),
      byTime,
      steps: s.errorSteps,
      error: s.error2.map((e) => Math.max(1e-4, e)),
    }
  }, [s])

  const xA = useAxis({ label: 'x', range: [0, TWO_PI] })
  const tA = useAxis({ label: 't', range: [0, 1] })
  const xB = useAxis({ label: 'x', range: [0, TWO_PI] })
  const tB = useAxis({ label: 't', range: [0, 1] })
  const sC = useAxis({ label: 'step', range: [0, Math.max(10, adamSteps + lbfgsSteps)] })
  const eC = useAxis({ label: 'relative error', log: true, range: [1e-4, 10] })
  const tD = useAxis({ label: 't', range: [0, 1] })
  const rD = useAxis({ label: 'mean squared residual', log: true })
  const percent = (e?: number) => (e === undefined || !Number.isFinite(e) ? '…' : `${(100 * e).toFixed(1)}%`)

  return (
    <Figure
      title="Convection breaks a physics-informed network"
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      caption={`The convection equation u_t + βu_x = 0 on [0, 2π] × [0, 1] from u(x, 0) = sin x, with the exact end values, whose solution sin(x − βt) is the initial wave moved at speed β. A network with two hidden layers of 16 tanh units (32 with the wide switch) is trained on the collocation points by Adam, then L-BFGS; the defaults take about two seconds, and the longer settings quoted in the text (wide, 256 points, 1000 Adam and 300 L-BFGS steps) about twenty. Move β to see the network fail as the wave crosses the domain more often. The switches add 16 random Fourier features of scale 2 to the inputs, causal weights exp(−10 · residual loss of the earlier time bins) over 16 bins, or a curriculum that raises β from 1 over the first 70% of the Adam steps. Top: the exact solution and the network, on one colour scale. Bottom left: the relative L2 error against the step. Bottom right: the network's mean squared residual at each time on the grid.`}
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
          <Readout label="relative error" value={percent(s?.error2.at(-1))} />
          <Readout label="loss" value={s ? (s.loss.at(-1) ?? NaN).toExponential(2) : '…'} />
          <Readout
            label="smallest causal weight"
            value={s?.causalWeights ? Math.min(...s.causalWeights).toFixed(3) : 'off'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={xA} y={tA} title="exact u(x, t)">
            {view && (
              <Raster x={view.xs} y={view.ts} z={view.exact} scale="diverging" range={[-1.2, 1.2]} valueLabel="u" />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xB} y={tB} title="network u_θ(x, t)">
            {view && (
              <Raster
                x={view.xs}
                y={view.ts}
                z={view.prediction}
                scale="diverging"
                range={[-1.2, 1.2]}
                valueLabel="u_θ"
              />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sC} y={eC} title="error against the exact solution">
            {view && <Curve name="relative error" x={view.steps} y={view.error} slot={0} emphasis />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={tD} y={rD} title="residual by time">
            {view && <Curve name="mean squared residual" x={view.ts} y={view.byTime} slot={0} />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
