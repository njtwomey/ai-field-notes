import { useMemo } from 'react'
import { Curve, Figure, float, int, Player, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { BoundaryMap, MOON_NAMES } from '../_shared/BoundaryMap'
import { sharpen } from 'aifn-methods/neural/semi-supervised'
import {
  byClass,
  classifier,
  consistencyRun,
  field,
  moonsProblem,
  MOONS_BOX,
  MOONS_GRID,
  percent,
  type RunSetup,
} from '../_shared/consistency-runs'
import { FixedHeight, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

const EVERY = 10
const PANEL = 290
const Q = Array.from({ length: 101 }, (_, k) => k / 100)

/** The average guess q̄ behind a two-class sharpened value s at temperature T (sharpening is invertible for K = 2). */
function unsharpen(s: number, T: number): number {
  const a = Math.pow(s, T)
  const b = Math.pow(1 - s, T)
  return a / (a + b)
}

/**
 * MixMatch on two moons with eight labels: the sharpening map from averaged guesses to targets, the targets of the
 * unlabelled points at a checkpoint, and the boundary trained with or without mixup.
 */
export function GuessAndSharpen() {
  const state = useFigureState({
    temperature: slider(0.05, 1, 0.5, { step: 0.05, label: 'sharpening temperature T' }),
    alpha: float(0.75, { ge: 0, le: 4, label: 'mixup α (0: no mixup)', suggestions: [0, 0.3, 0.75] }),
    copies: int(2, { ge: 1, le: 8, label: 'augmentations per point K′', suggestions: [1, 2, 4] }),
    weight: float(30, { ge: 0, le: 200, label: 'unlabelled weight λ_U', suggestions: [10, 30, 75] }),
    steps: int(600, { ge: 10, le: 2000, label: 'training steps', suggestions: [200, 600, 1200] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { temperature, alpha, copies, weight, steps, rate, seed } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'mixmatch', temperature, alpha, augmentations: copies },
      steps,
      rate,
      weight,
      rampUp: Math.round(steps / 3),
      weak: 0.1,
      strong: 0.1,
      seed,
      every: EVERY,
    }),
    [problem, temperature, alpha, copies, steps, rate, weight, seed],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])

  const view = useMemo(() => {
    if (!cp) return null
    const values = field(net, cp.state.params, MOONS_GRID, 2)
    const has = cp.step > 0
    const guessed = Int32Array.from(cp.state.pseudo, (c) => (has ? c : -1))
    // Each point's averaged guess and sharpened target for class "inner moon".
    const target = Array.from({ length: problem.unlabelled.length }, (_, i) => cp.state.targets[2 * i + 1])
    const average = target.map((s) => unsharpen(s, temperature))
    let confident = 0
    target.forEach((s) => (confident += Math.max(s, 1 - s) > 0.95 ? 1 : 0))
    return { values, guessed, target, average, has, confident: confident / target.length }
  }, [cp, net, problem, temperature])

  const curve = useMemo(
    () =>
      Array.from(
        sharpen(
          Q.flatMap((q) => [1 - q, q]),
          2,
          temperature,
        ),
      ).filter((_, j) => j % 2 === 1),
    [temperature],
  )
  const qa = useAxis({ label: 'averaged guess q̄ (inner moon)', range: [0, 1] })
  const ta = useAxis({ label: 'sharpened target', range: [0, 1], equal: qa })

  return (
    <Figure
      title="Guessing and sharpening labels in MixMatch"
      state={state}
      defaultSize="XL"
      caption={`Two moons with 8 labelled points (large marks) and 192 unlabelled ones; two hidden layers of 32 tanh units, trained full batch with Adam. At every step each unlabelled point gets K′ copies with Gaussian jitter of sd 0.1; the network's predictions on the copies are averaged into a guess q̄ and sharpened with temperature T into the target. Labelled and unlabelled copies are then shuffled together and mixed pairwise with mixup weight λ′ = max(λ, 1 − λ), λ ~ Beta(α, α); the loss is the cross-entropy on the mixed labelled copies plus λ_U times the squared error on the mixed unlabelled ones. λ_U ramps up over the first third of training. Left: the boundary at the player's checkpoint and each unlabelled point coloured by the class of its target. Right: the sharpening map for the chosen T (curve), with the unlabelled points' guesses and targets at the checkpoint; T = 1 is the identity (dashed). On this two-dimensional problem the run with α = 0 and T = 1 does best; the note explains why.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="accuracy on unlabelled points" value={percent(cp?.accuracy)} />
          <Readout label="targets with the right class" value={percent(cp?.state.pseudoAccuracy)} />
          <Readout label="targets above 0.95" value={percent(view?.has ? view.confident : undefined)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={MOONS_GRID}
            box={MOONS_BOX}
            values={view?.values ?? null}
            hideUnlabelled={Boolean(view?.has)}
            title="targets on the plane"
          >
            {view?.has &&
              [0, 1].map((c) => (
                <Points
                  key={c}
                  name={`target: ${MOON_NAMES[c]}`}
                  {...byClass(problem.unlabelled, view.guessed, c)}
                  slot={c}
                  thin
                />
              ))}
          </BoundaryMap>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={qa} y={ta} title="sharpening">
            <Curve name="no sharpening (T = 1)" x={[0, 1]} y={[0, 1]} muted dashed silent />
            <Curve name={`sharpening at T = ${temperature}`} x={Q} y={curve} emphasis />
            {view?.has && <Points name="unlabelled points" x={view.average} y={view.target} slot={2} thin />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
