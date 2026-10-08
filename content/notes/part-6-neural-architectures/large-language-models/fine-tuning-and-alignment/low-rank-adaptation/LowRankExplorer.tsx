import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Annotation,
  Bars,
  call,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  when,
} from 'aifn-render'
import type { LowRankFitOptions, LowRankFitResult, LowRankTargetOptions } from 'aifn-methods/neural/adaptation'

const MODULE = 'neural/adaptation'
/** The side of the square target ΔW. */
const SIZE = 64
/** The scale's numerator α, fixed so that only the convention (α/r or α/√r) changes s. */
const ALPHA = 16
/** About this many checkpoints per run, for the player. */
const CHECKPOINTS = 100
const HEIGHT = 300

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Landed = { fit: LowRankFitResult; key: string }

/** One fit of the shared target at one rank, in the compute worker: a rank-64 run takes most of a second. */
function useFit(target: LowRankTargetOptions, options: Omit<LowRankFitOptions, 'rank'>, rank: number, key: string) {
  return useComputed(
    () => call<LowRankFitResult>(`${MODULE}/lowRankFit`, call(`${MODULE}/lowRankTarget`, target), { ...options, rank }),
    [target, options, rank],
    {
      mode: 'worker',
      initial: null as Landed | null,
      then: (fit): Landed => ({ fit, key: `${key}|${rank}` }),
      cancelAfter: 400,
    },
  )
}

export function LowRankExplorer() {
  const state = useFigureState({
    r: slider(1, SIZE, 16, { step: 1, label: 'rank r' }),
    scale: choice(
      [
        { value: 'inverse', label: 'α/r (LoRA)' },
        { value: 'inverse-sqrt', label: 'α/√r (rsLoRA)' },
      ],
      'inverse',
      { label: 'scale s' },
    ),
    optimiser: choice(
      [
        { value: 'sgd', label: 'gradient descent' },
        { value: 'adam', label: 'Adam' },
      ],
      'sgd',
      { label: 'optimiser' },
    ),
    sgdRate: float(0.05, {
      gt: 0,
      le: 1,
      scale: 'log10',
      suggestions: [0.01, 0.02, 0.05, 0.1],
      label: 'learning rate η',
      when: when('optimiser', 'sgd'),
    }),
    adamRate: float(0.005, {
      gt: 0,
      le: 0.1,
      scale: 'log10',
      suggestions: [0.001, 0.002, 0.005, 0.01, 0.02],
      label: 'learning rate η',
      when: when('optimiser', 'adam'),
    }),
    init: choice(
      [
        { value: 'lora', label: 'LoRA: B = 0, A random' },
        { value: 'pissa', label: 'PiSSA: top-r singular directions' },
      ],
      'lora',
      { label: 'start' },
    ),
    spectrum: choice(
      [
        { value: 'power', label: 'power law σᵢ = i^−d' },
        { value: 'low-rank-plus-noise', label: 'rank 4 plus noise' },
      ],
      'power',
      { label: 'target spectrum' },
    ),
    decay: float(1, {
      ge: 0,
      le: 3,
      suggestions: [0, 0.5, 1, 2],
      label: 'decay d',
      when: when('spectrum', 'power'),
    }),
    steps: int(300, { ge: 10, le: 1000, suggestions: [100, 300, 1000], label: 'steps' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    logLoss: setting(false, 'log loss axis'),
  })
  const { scale, optimiser, sgdRate, adamRate, init, spectrum, decay, steps, seed, logLoss } = state
  const r = Math.round(state.r)
  const learningRate = optimiser === 'sgd' ? sgdRate : adamRate

  const target = useMemo<LowRankTargetOptions>(
    () => ({ size: SIZE, decay: spectrum === 'power' ? decay : 'low-rank-plus-noise', seed }),
    [spectrum, decay, seed],
  )
  const options = useMemo<Omit<LowRankFitOptions, 'rank'>>(
    () => ({
      scale,
      alpha: ALPHA,
      optimiser,
      learningRate,
      steps,
      init,
      seed,
      every: Math.max(1, Math.round(steps / CHECKPOINTS)),
    }),
    [scale, optimiser, learningRate, steps, init, seed],
  )
  const key = `${JSON.stringify(target)}|${JSON.stringify(options)}`

  // The chosen rank and the ranks a factor of 4 either side, each in a fixed slot.
  const lower = Math.max(1, Math.round(r / 4))
  const higher = Math.min(SIZE, 4 * r)
  const fits = [
    useFit(target, options, lower, key),
    useFit(target, options, r, key),
    useFit(target, options, higher, key),
  ]
  const roles = [
    { rank: lower, slot: 0, shown: lower < r },
    { rank: r, slot: 1, shown: true },
    { rank: higher, slot: 2, shown: higher > r },
  ]
  const runs = roles.flatMap((role, k) => {
    const landed = fits[k].value
    return role.shown && landed ? [{ ...role, fit: landed.fit, stale: landed.key !== `${key}|${role.rank}` }] : []
  })
  const main = runs.find((run) => run.slot === 1)

  const [pos, setPos] = useState({ key, step: 0 })
  const count = main ? main.fit.checkpoints.length : 1
  const index = pos.key === key ? Math.min(pos.step, count - 1) : 0
  const atStep = main?.fit.checkpoints[index]?.step ?? 0

  const stepAxis = useMemo(() => Array.from({ length: steps + 1 }, (_, t) => t), [steps])
  const indexAxis = useMemo(() => Array.from({ length: SIZE }, (_, i) => i + 1), [])

  const lx = useAxis({ label: 'step', integer: true, range: [0, steps] })
  const ly = useAxis({ label: '½‖sBA − ΔW‖²', log: logLoss, range: logLoss ? undefined : [0, undefined], key })
  const sx = useAxis({ label: 'singular value index i', range: [0.5, SIZE + 0.5] })
  const sy = useAxis({ label: 'singular value', range: [0, undefined], key: target })

  const name = (rank: number) => `r = ${rank}`
  return (
    <Figure
      title="LoRA rank explorer"
      state={state}
      defaultSize="L"
      caption={`A target change ΔW (${SIZE} × ${SIZE}, random singular vectors, the chosen spectrum) is fitted by s·B·A, with B ${SIZE} × r and A r × ${SIZE}, by minimising ½‖sBA − ΔW‖² from LoRA's start (B = 0) or PiSSA's (the target's best rank-r approximation), with α = ${ALPHA}. Three ranks run at once: r and the ranks a factor of 4 either side. Left: the loss by step, each rank's Eckart–Young floor, ½ Σ σᵢ² over i > r, dashed in its colour; no rank-r update gets below its floor. Right: the target's singular values (grey) and those of the learned update sBA at the played step (one curve per rank). Drag the rank guide on the right, or set r. Play the steps: from B = 0 each rank fills in the target's top r singular values, largest first, and then stops; the values past r stay at zero. With the default settings and gradient descent, α/r slows each larger rank: rank 64 takes about six times as many steps as rank 4 to close half of its gap to its floor. α/√r brings the three within a factor of about three. At step 300, α/r leaves rank 64 above rank 16; α/√r puts it lowest. With Adam the conventions trade places: α/r keeps ranks 16 and 64 together and rank 4 within a factor of two, while α/√r lets rank 64 close half its gap about six times as fast as rank 4. PiSSA starts every rank at its floor. Gradient descent stays there. Adam first rises a little off it, because it scales the rounding-level gradient up to a step of the learning rate, and then returns.`}
      controls={
        <Player
          value={index}
          onChange={(s) => setPos({ key, step: s })}
          count={count}
          label="step"
          format={(s) => `step ${main?.fit.checkpoints[s]?.step ?? 0} of ${steps}`}
        />
      }
      readouts={
        <>
          {runs.map((run) => (
            <Readout
              key={run.slot}
              label={`${name(run.rank)}: loss at step ${atStep} (floor)`}
              value={`${formatNumber(run.fit.losses[atStep] ?? NaN)} (${formatNumber(run.fit.floor)})`}
            />
          ))}
          {main && <Readout label={`scale s at r = ${r}`} value={formatNumber(main.fit.scale)} />}
        </>
      }
    >
      {!main ? (
        <p className="py-12 text-center text-sm text-muted-foreground">Fitting…</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <FixedHeight height={HEIGHT}>
            <Plot x={lx} y={ly}>
              {runs.map((run) => (
                <Curve
                  key={`loss-${run.slot}`}
                  id={`loss-${run.slot}`}
                  name={name(run.rank)}
                  x={stepAxis}
                  y={run.fit.losses}
                  slot={run.slot}
                  stale={run.stale}
                />
              ))}
              {runs.map((run) =>
                run.fit.floor > 0 || !logLoss ? (
                  <Annotation
                    key={`floor-${run.slot}`}
                    id={`floor-${run.slot}`}
                    name={name(run.rank)}
                    y={run.fit.floor}
                    slot={run.slot}
                    dashed
                  />
                ) : null,
              )}
              <Annotation x={atStep} muted live />
              <Points
                name="at the played step"
                x={runs.map(() => atStep)}
                y={runs.map((run) => run.fit.losses[atStep] ?? NaN)}
                emphasis
                size={7}
                live
              />
            </Plot>
          </FixedHeight>
          <FixedHeight height={HEIGHT}>
            <Plot x={sx} y={sy}>
              <Bars name="target ΔW" x={indexAxis} y={main.fit.targetSpectrum} width={0.7} muted />
              {[...runs].reverse().map((run) => {
                const values = run.fit.checkpoints[index]?.spectrum ?? new Float64Array(run.rank)
                return (
                  <Curve
                    key={`spectrum-${run.slot}`}
                    id={`spectrum-${run.slot}`}
                    name={`update, ${name(run.rank)}`}
                    x={indexAxis.slice(0, values.length)}
                    y={values}
                    slot={run.slot}
                    showPoints={run.slot === 1}
                    width={1.5}
                    stale={run.stale}
                  />
                )
              })}
              <Handle {...state.handle('r', { label: `rank r = ${r}` })} />
            </Plot>
          </FixedHeight>
        </div>
      )}
    </Figure>
  )
}
