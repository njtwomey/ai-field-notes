/**
 * `GymTrainer`: train any agent in any environment headlessly in the compute worker (`aifn-applied/gym` `training`),
 * watch the learning curve fill in as summaries stream, pick an episode on it (drag or click the marker), and play that
 * episode step by step: `replay` re-runs the training episode exactly as it happened (from the nearest checkpoint),
 * `evaluate` plays a fresh greedy episode of the policy as it was after that episode. The environment is drawn by the
 * renderer for its `render.kind` (`GYM_RENDERERS`).
 *
 * The page owns the controls (episodes, seed, the agent's hyperparameters, in its `useFigureState`) and passes a
 * `setup`: the environment and agent built on the page (for replay and evaluation) and the same two as worker tasks
 * (for training). Training starts on mount; afterwards it runs when Train is pressed, so moving a control does not
 * discard a run until asked.
 */
import { useMemo, useState, type ReactNode } from 'react'
import { evaluateEpisode, replay, type Training, type Trajectory } from 'aifn-applied/gym'
import type { Agent, Environment } from 'aifn/foundation/contracts'
import { Player } from '@lab/controls'
import { ControlRow, Figure, type FigureProps } from '@lab/layout'
import { call, useStreamed, type Task } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { GYM_RENDERERS, renderKind, WIDE_KINDS } from './registry'
import type { GymRenderer } from './renderers'
import { hasStepSeries } from './series'
import { StepSeries } from './StepSeries'

/** What to train: the environment and agent on the page, the same as worker tasks, and the run's length and seed. */
export type GymSetup = {
  env: Environment<unknown, unknown, unknown>
  agent: Agent<unknown, unknown, unknown>
  /** `call('gym/environments/<key>', params)` for the worker. */
  envTask: Task
  /** `call('gym/agents/<key>', params)`. */
  agentTask: Task
  episodes: number
  seed: number | string
  /** The environment `evaluate` plays in, when it differs from the trained one (another start state). Not trained on. */
  evaluationEnv?: Environment<unknown, unknown, unknown>
}

export type GymTrainerProps = {
  title: string
  purpose: ReactNode
  caption?: ReactNode
  /** The page's figure state (its controls appear above the trainer's). */
  state?: FigureProps['state']
  setup: GymSetup
  /** A renderer for this page, instead of the one for `render.kind`. */
  renderer?: GymRenderer
  /** The evaluation episode's seed. Default 'evaluate'. */
  evaluationSeed?: number | string
  defaultSize?: 'L' | 'XL'
  /** The mode an episode opens in. Default 'replay'. */
  initialMode?: 'replay' | 'evaluate'
  /** Options passed to the renderer (`GymRenderProps.options`). */
  rendererOptions?: Readonly<Record<string, unknown>>
}

type Mode = 'replay' | 'evaluate'

/** A trailing moving average over `1 / fraction` of the run (at least 1 episode). */
function smooth(y: ArrayLike<number>, total: number, fraction: number): number[] {
  const w = Math.max(1, Math.round(total * fraction))
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < y.length; i++) {
    sum += y[i]
    if (i >= w) sum -= y[i - w]
    out.push(sum / Math.min(i + 1, w))
  }
  return out
}

const keyOf = (s: GymSetup) => JSON.stringify([s.envTask, s.agentTask, s.episodes, s.seed])

export function GymTrainer({
  title,
  purpose,
  caption,
  state,
  setup,
  renderer,
  evaluationSeed = 'evaluate',
  defaultSize = 'XL',
  initialMode = 'replay',
  rendererOptions,
}: GymTrainerProps) {
  // The setup last trained: the first one on mount, then whatever is current when Train is pressed.
  const [trained, setTrained] = useState(setup)
  const stale = keyOf(trained) !== keyOf(setup)
  const task = useMemo(
    () =>
      call<Training<unknown>>('gym/training', trained.envTask, trained.agentTask, {
        episodes: trained.episodes,
        seed: trained.seed,
      }),
    [trained],
  )
  const run = useStreamed(task)
  const training = run.value
  const n = training?.episodes ?? 0

  // The chosen episode follows the newest one until the reader picks one.
  // A pick belongs to the run it was made on; a new run starts from its newest episode again.
  const [picked, setPicked] = useState<{ task: typeof task; episode: number } | null>(null)
  const episode = Math.min(picked?.task === task ? picked.episode : n, n)
  const [mode, setMode] = useState<Mode>(initialMode)
  const evaluationEnv = setup.evaluationEnv ?? trained.env
  const shownEnv = mode === 'evaluate' ? evaluationEnv : trained.env

  const trajectory = useMemo((): Trajectory<unknown, unknown, unknown> | null => {
    if (!training || episode < 1) return null
    const { env, agent } = trained
    return mode === 'replay'
      ? replay(env, agent, training, episode).trajectory
      : evaluateEpisode(evaluationEnv, agent, training, episode, evaluationSeed).trajectory
  }, [training, episode, mode, trained, evaluationSeed, evaluationEnv])
  // The step belongs to the trajectory it was set on: a newly chosen episode opens at step 0.
  const [stepOf, setStepOf] = useState<{ trajectory: typeof trajectory; step: number } | null>(null)
  const step = stepOf?.trajectory === trajectory ? stepOf.step : 0
  const setStep = (s: number) => setStepOf({ trajectory, step: s })
  const steps = trajectory?.states.length ?? 1
  const at = Math.min(step, steps - 1)

  const curves = useMemo(() => {
    if (!training) return null
    const x = Array.from({ length: training.episodes }, (_, i) => i + 1)
    const scalar = Object.entries(training.scalars)[0]
    const oneStep = training.lengths.every((l) => l === 1)
    let cum = 0
    // Episodes by outcome, and the share of failures over a trailing window, when the environment reports outcomes.
    const outcome = training.outcome
    const pick = (o: number) => x.filter((e) => outcome[e - 1] === o)
    const failX = pick(-1)
    const okX = pick(1)
    const w = Math.max(1, Math.round(training.total / 20))
    let fails = 0
    const failShare = Array.from(outcome, (o, i) => {
      fails += o === -1 ? 1 : 0
      if (i >= w && outcome[i - w] === -1) fails -= 1
      return fails / Math.min(i + 1, w)
    })
    return {
      x,
      // One-step episodes (a bandit's rounds) pay noisy single rewards: no raw curve, and a wider average.
      returns: oneStep ? null : Array.from(training.returns),
      smoothed: smooth(training.returns, training.total, oneStep ? 1 / 10 : 1 / 50),
      lengths: Array.from(training.lengths),
      regret: training.regret ? Array.from(training.regret, (r) => (cum += r)) : null,
      scalar: scalar ? { name: scalar[0], y: Array.from(scalar[1]) } : null,
      outcomes:
        failX.length + okX.length > 0
          ? {
              fail: { x: failX, y: failX.map((e) => training.returns[e - 1]) },
              ok: { x: okX, y: okX.map((e) => training.returns[e - 1]) },
              share: failShare,
              window: w,
            }
          : null,
    }
  }, [training])

  const kind = renderKind(trained.env)
  const Renderer = renderer ?? GYM_RENDERERS[kind]
  // Per-step panels (state series and actions against step) for environments that declare state series.
  const withSeries = hasStepSeries(trained.env, kind)
  const wide = WIDE_KINDS.has(kind)
  const total = trained.episodes
  const ea = useAxis({ label: 'episode', range: [0, total], key: total })
  const ra = useAxis({ label: 'return', hold: 'union', key: task })
  const ba = useAxis({ label: curves?.regret ? 'cumulative regret' : 'length (steps)', hold: 'union', key: task })
  const middle = curves?.outcomes
    ? `failure share (last ${curves.outcomes.window})`
    : (curves?.scalar?.name ?? 'reward')
  const sa = useAxis({
    label: middle,
    hold: 'union',
    key: task,
    ...(curves?.outcomes && { range: [0, 1] as const }),
  })
  const sx = useAxis({ label: 'step' })
  const pick = (v: number) => setPicked({ task, episode: Math.max(1, Math.min(n, Math.round(v))) })
  const marker = n > 0 && <Handle kind="x" at={episode} label="episode" onDrag={pick} />

  const rewardsSoFar = trajectory ? trajectory.rewards.slice(0, at).reduce((a, b) => a + b, 0) : 0
  // At the last step of an episode that the environment calls a success or a failure, the scene takes that tone.
  const ending = trajectory?.ending ?? null
  const end = ending && at === steps - 1 ? ending : null
  const T = steps - 1
  const outcomeText = !trajectory
    ? '—'
    : ending
      ? ending.success
        ? /\bsteps?\b/.test(ending.reason)
          ? ending.reason
          : `${ending.reason} at step ${T}`
        : `failed at step ${T}: ${ending.reason}`
      : trajectory.reachedTerminal
        ? `terminal at step ${T}`
        : `truncated at step ${T}`
  const scene =
    training && trajectory && Renderer ? (
      <Renderer
        env={shownEnv}
        trajectory={trajectory}
        step={at}
        training={training}
        episode={episode}
        options={rendererOptions}
        end={end}
      />
    ) : (
      <Plot x={sx} y={sa} title="waiting for the first episodes" />
    )
  const returnPlot = (
    <Plot x={ea} y={ra} title="return per episode">
      {curves?.returns && <Curve name="return" x={curves.x} y={curves.returns} muted thin />}
      {curves && <Curve name="moving average" x={curves.x} y={curves.smoothed} slot={0} />}
      {curves?.outcomes && (
        <Points name="failed" x={curves.outcomes.fail.x} y={curves.outcomes.fail.y} tone="destructive" thin />
      )}
      {curves?.outcomes && (
        <Points name="succeeded" x={curves.outcomes.ok.x} y={curves.outcomes.ok.y} tone="success" thin />
      )}
      {marker}
    </Plot>
  )
  const middlePlot = (
    <Plot x={curves?.outcomes || curves?.scalar ? ea : sx} y={sa} legend={false}>
      {curves?.outcomes ? (
        <Curve name={middle} x={curves.x} y={curves.outcomes.share} tone="destructive" />
      ) : curves?.scalar ? (
        <Curve name={curves.scalar.name} x={curves.x} y={curves.scalar.y} slot={2} />
      ) : (
        trajectory && (
          <Curve
            name="reward per step"
            x={trajectory.rewards.map((_, i) => i + 1)}
            y={trajectory.rewards}
            slot={2}
            showPoints
          />
        )
      )}
      {(curves?.outcomes || curves?.scalar) && marker}
    </Plot>
  )
  const lastPlot = (
    <Plot x={ea} y={ba} legend={false}>
      {curves && (
        <Curve
          name={curves.regret ? 'cumulative regret' : 'length'}
          x={curves.x}
          y={curves.regret ?? curves.lengths}
          slot={3}
        />
      )}
      {marker}
    </Plot>
  )
  return (
    <Figure
      title={title}
      purpose={purpose}
      state={state}
      defaultSize={defaultSize}
      controls={
        <>
          <ControlRow label="train">
            <div className="flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                variant={stale ? 'default' : 'outline'}
                aria-label="Train"
                onClick={() => setTrained(setup)}
              >
                Train
              </Button>
              <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
                <div className="h-full bg-primary" style={{ width: `${(100 * n) / Math.max(1, total)}%` }} />
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">
                {run.error
                  ? `failed: ${run.error}`
                  : `${n} / ${total} episodes${run.running ? '…' : ''}${stale ? ' · settings changed: press Train' : ''}`}
              </span>
            </div>
          </ControlRow>
          <ControlRow label={`episode ${episode}`}>
            <div className="flex items-center gap-1">
              {(['replay', 'evaluate'] as const).map((m) => (
                <Button
                  key={m}
                  size="sm"
                  variant={mode === m ? 'default' : 'outline'}
                  aria-label={m}
                  aria-pressed={mode === m}
                  onClick={() => setMode(m)}
                >
                  {m === 'replay' ? 'replay (as trained)' : 'evaluate (greedy)'}
                </Button>
              ))}
            </div>
            <Player label="step" value={at} onChange={setStep} count={steps} defaultSpeed={10} />
          </ControlRow>
          {end && (
            <div
              role="status"
              className={
                end.success
                  ? 'border-success/40 bg-success/10 text-success col-span-full w-fit rounded-md border px-3 py-1.5 text-sm'
                  : 'col-span-full w-fit rounded-md border border-destructive/40 bg-destructive/10 px-3 py-1.5 text-sm text-destructive'
              }
            >
              {end.success ? '✓ ' : '✗ '}
              {outcomeText}
            </div>
          )}
        </>
      }
      readouts={
        <>
          <Readout label="episode" value={episode} />
          <Readout label="step" value={`${at} / ${steps - 1}`} />
          <Readout label="reward so far" value={rewardsSoFar.toFixed(2)} />
          <Readout label="episode return" value={trajectory ? trajectory.episodeReturn.toFixed(2) : '—'} />
          <Readout label="outcome" value={outcomeText} />
        </>
      }
      caption={
        <>
          {caption} Trained headlessly in the worker by aifn <code>training</code> ({total} episodes, seed{' '}
          {String(trained.seed)}, checkpoints every {training?.every ?? '…'} episodes); drag or click the episode marker
          to pick an episode, then play it: replay re-runs the training episode from the nearest checkpoint, evaluate
          plays the greedy policy as of that episode on seed {String(evaluationSeed)}.
        </>
      }
    >
      {wide ? (
        <>
          <Plots cols={1} scale={withSeries ? 0.26 : 0.45}>
            {scene}
          </Plots>
          <Plots cols={3} scale={withSeries ? 0.36 : 0.55}>
            {returnPlot}
            {middlePlot}
            {lastPlot}
          </Plots>
        </>
      ) : (
        <Plots cols={2} rows={2} heights={[2.2, 1]} widths={[1, 1.3]} scale={withSeries ? 0.62 : 1}>
          {scene}
          {returnPlot}
          {middlePlot}
          {lastPlot}
        </Plots>
      )}
      {withSeries && trajectory && (
        <StepSeries env={shownEnv} kind={kind} trajectory={trajectory} step={at} onStep={setStep} scale={0.38} />
      )}
    </Figure>
  )
}
