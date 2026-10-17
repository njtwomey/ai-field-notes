import { useMemo } from 'react'
import { Curve, Figure, float, int, Player, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { BoundaryMap } from '../_shared/BoundaryMap'
import {
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

const EVERY = 2
const PANEL = 260

/** Mean and standard deviation of a list. */
function spread(v: number[]): { mean: number; sd: number } {
  if (v.length === 0) return { mean: NaN, sd: NaN }
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  return { mean, sd: Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length) }
}

/**
 * Mean teacher on two moons with eight labels: the student's and the teacher's decision boundaries at a checkpoint,
 * and their accuracies over the whole run, for a chosen EMA decay.
 */
export function TeacherStudent() {
  const state = useFigureState({
    decay: float(0.97, { ge: 0, le: 0.9999, label: 'EMA decay α', suggestions: [0, 0.9, 0.97, 0.99, 0.999] }),
    weight: float(30, { ge: 0, le: 200, label: 'consistency weight λ', suggestions: [10, 30, 100] }),
    sigma: float(0.1, { ge: 0, le: 0.5, label: 'input noise sd σ', suggestions: [0.05, 0.1, 0.2] }),
    steps: int(600, { ge: 10, le: 2000, label: 'training steps', suggestions: [200, 600, 1200] }),
    rate: float(0.01, { gt: 0, le: 0.1, label: 'Adam step size', suggestions: [0.003, 0.01, 0.03] }),
    seed: int(2, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { decay, weight, sigma, steps, rate, seed } = state
  const problem = useMemo(() => moonsProblem(seed), [seed])
  const setup = useMemo<RunSetup>(
    () => ({
      problem,
      method: { kind: 'mean-teacher', decay },
      steps,
      rate,
      weight,
      rampUp: Math.round(steps / 3),
      weak: sigma,
      strong: sigma,
      seed,
      every: EVERY,
    }),
    [problem, decay, steps, rate, weight, sigma, seed],
  )
  const run = useSlicedRun(setup, consistencyRun)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const net = useMemo(() => classifier(2), [])
  const maps = useMemo(
    () =>
      cp
        ? {
            student: field(net, cp.state.params, MOONS_GRID, 2),
            teacher: field(net, cp.state.teacher ?? cp.state.params, MOONS_GRID, 2),
          }
        : null,
    [cp, net],
  )
  const history = useMemo(() => {
    const all = run.checkpoints
    const half = all.filter((c) => c.step > steps / 2)
    return {
      step: all.map((c) => c.step),
      student: all.map((c) => c.accuracy),
      teacher: all.map((c) => c.teacherAccuracy),
      student2: spread(half.map((c) => c.accuracy)),
      teacher2: spread(half.map((c) => c.teacherAccuracy)),
    }
  }, [run.checkpoints, steps])

  const sa = useAxis({ label: 'step', range: [0, steps] })
  const aa = useAxis({ label: 'accuracy on unlabelled points', range: [0.5, 1] })
  const stepNow = cp?.step ?? 0
  const sd = (v: number) => (Number.isFinite(v) ? (100 * v).toFixed(2) : '…')

  return (
    <Figure
      title="A mean teacher on two moons"
      state={state}
      defaultSize="XL"
      caption={`Two moons with 8 labelled points (large marks) and 192 unlabelled ones; two hidden layers of 32 tanh units, trained full batch with Adam. The student is trained on the labelled cross-entropy plus λ times the squared difference between its softmax output on one noisy copy of each unlabelled point and the teacher's output on another noisy copy (Gaussian noise of sd σ). After every update the teacher's weights become α times their old value plus 1 − α times the student's; the gradient never flows into the teacher. λ ramps up over the first third of training. Top: the student's and the teacher's boundaries at the player's checkpoint. Bottom: both networks' accuracy on the unlabelled points at every second step; the vertical line marks the checkpoint. With α = 0 the teacher is the student and the run is a Π-model with the gradient through one pass. Larger α smooths the teacher's accuracy over steps and makes it lag behind the student.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={{
        'at the checkpoint': (
          <>
            <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
            <Readout label="student accuracy" value={percent(cp?.accuracy)} />
            <Readout label="teacher accuracy" value={percent(cp?.teacherAccuracy)} />
          </>
        ),
        'second half of the run': (
          <>
            <Readout label="student mean" value={percent(history.student2.mean)} />
            <Readout label="student sd (points)" value={sd(history.student2.sd)} />
            <Readout label="teacher mean" value={percent(history.teacher2.mean)} />
            <Readout label="teacher sd (points)" value={sd(history.teacher2.sd)} />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={MOONS_GRID}
            box={MOONS_BOX}
            values={maps?.student ?? null}
            title="student θ"
          />
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <BoundaryMap
            problem={problem}
            grid={MOONS_GRID}
            box={MOONS_BOX}
            values={maps?.teacher ?? null}
            title="teacher θ′ (EMA of the student)"
          />
        </FixedHeight>
      </div>
      <FixedHeight height={PANEL}>
        <Plot x={sa} y={aa} title="accuracy during training">
          <Curve name="student" x={history.step} y={history.student} slot={0} />
          <Curve name="teacher" x={history.step} y={history.teacher} slot={1} />
          <Curve name="checkpoint" x={[stepNow, stepNow]} y={[0.5, 1]} muted dashed silent />
        </Plot>
      </FixedHeight>
    </Figure>
  )
}
