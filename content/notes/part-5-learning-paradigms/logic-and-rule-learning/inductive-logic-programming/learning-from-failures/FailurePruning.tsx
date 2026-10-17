import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { learnFromFailures, type LffConstraint } from 'aifn-compute/logic/induction'
import { ANCESTOR_TASK, LAST_TASK } from './tasks'

const TASKS = { last: LAST_TASK, ancestor: ANCESTOR_TASK } as const

const schema = {
  task: choice(
    [
      { value: 'last', label: LAST_TASK.label },
      { value: 'ancestor', label: ANCESTOR_TASK.label },
    ] as const,
    'last',
    { label: 'task' },
  ),
  generalisation: setting(true, 'generalisation constraints'),
  specialisation: setting(true, 'specialisation constraints'),
  elimination: setting(true, 'elimination constraints'),
}

const SLOT: Record<LffConstraint, number> = { generalisation: 0, specialisation: 1, elimination: 2 }
const WHY: Record<LffConstraint, string> = {
  generalisation: 'too general: prune its generalisations',
  specialisation: 'too specific: prune its specialisations',
  elimination: 'proves no positive: prune non-recursive programs containing a specialisation of it',
}

/** Popper's generate, test and constrain loop over a space of programs of one or two clauses. */
export function FailurePruning() {
  const state = useFigureState(schema)
  const key = `${state.task}/${state.generalisation}/${state.specialisation}/${state.elimination}`
  const run = useMemo(
    () =>
      learnFromFailures(TASKS[state.task].problem, TASKS[state.task].bias, {
        generalisation: state.generalisation,
        specialisation: state.specialisation,
        elimination: state.elimination,
        maxTests: 2000,
      }),
    [state.task, state.generalisation, state.specialisation, state.elimination],
  )
  const [pos, setPos] = useState({ key, step: 0 })
  const last = run.steps.length - 1
  const t = pos.key === key ? Math.min(pos.step, last) : 0
  const step = run.steps[t]
  const task = TASKS[state.task].problem

  // Program {i} sits at (i, i), program {i, j} at (i, j).
  const view = useMemo(() => {
    const at = (p: number) => {
      const c = run.programs[p].clauses
      return [c[0], c[c.length - 1]] as const
    }
    const groups: Record<'open' | 'tested' | LffConstraint, number[]> = {
      open: [],
      tested: [],
      generalisation: [],
      specialisation: [],
      elimination: [],
    }
    run.programs.forEach((_, p) => {
      if (run.testedAt[p] >= 0 && run.testedAt[p] < t) groups.tested.push(p)
      else if (run.prunedAt[p] >= 0 && run.prunedAt[p] <= t) groups[run.prunedBy[p]!].push(p)
      else if (run.testedAt[p] !== t) groups.open.push(p)
    })
    const pts = (ids: number[]) => ({ x: ids.map((p) => at(p)[0]), y: ids.map((p) => at(p)[1]) })
    return { groups, pts, current: at(step.program) }
  }, [run, t, step])

  const n = run.clauses.length
  const xa = useAxis({ label: 'first clause (smallest first)', range: [-1, n], nice: false, integer: true })
  const ya = useAxis({ label: 'second clause', range: [-1, n], nice: false, integer: true, equal: xa })
  const xb = useAxis({ label: 'programs tested', range: [0, run.enumerateTests], integer: true })
  const yb = useAxis({ label: 'programs left', range: [0, run.programs.length] })
  const program = run.programs[step.program].clauses.map((i) => run.clauses[i].text)
  const proved = (bs: readonly boolean[]) => bs.filter(Boolean).length
  const tests = run.steps.slice(0, t + 1)

  return (
    <Figure
      title="Pruning the program space with each failure"
      state={state}
      defaultSize="L"
      controls={
        <Player
          label="test"
          value={t}
          onChange={(k) => setPos({ key, step: k })}
          count={run.steps.length}
          format={(k) => `${k + 1} of ${last + 1}`}
        />
      }
      equation={
        <div className="grid gap-2 text-left text-sm md:grid-cols-2">
          <pre className="font-mono text-xs leading-relaxed">{program.join('\n')}</pre>
          <div className="font-prose text-[15px] leading-snug">
            {step.solution
              ? 'Proves every positive and no negative: the solution.'
              : step.constraints.map((c) => WHY[c]).join('; ') || 'Fails, but every constraint is off.'}
          </div>
        </div>
      }
      readouts={{
        [`test ${t + 1}`]: (
          <>
            <Readout label="positives proved" value={`${proved(step.positives)} of ${task.positives.length}`} />
            <Readout label="negatives proved" value={`${proved(step.negatives)} of ${task.negatives.length}`} />
            <Readout
              label="newly pruned"
              value={`${step.pruned.generalisation} generalisations, ${step.pruned.specialisation} specialisations, ${step.pruned.elimination} eliminated`}
            />
          </>
        ),
        space: (
          <>
            <Readout label="clauses, programs" value={`${n}, ${run.programs.length}`} />
            <Readout label="programs left untested" value={step.remaining} />
            <Readout
              label="tests to the solution"
              value={
                run.solution >= 0
                  ? `${run.steps.length} with these constraints, ${run.enumerateTests} by plain generate and test`
                  : 'no solution in the space'
              }
            />
          </>
        ),
      }}
      caption={`Each point is a program: one clause (on the diagonal) or two clauses (above it), with the ${n} clauses of the space ordered from fewest literals to most. The learner tests the smallest program not yet tested or pruned (large ink point), and each failure prunes programs: generalisations of a program that proves a negative example (first colour), specialisations of one that misses a positive (second colour), and, when it proves no positive at all, every non-recursive program that contains a specialisation of one of its clauses (third colour). Small ink points were tested earlier; grey points are still open. Pick a task, switch constraint types off to see how many more tests plain generate and test needs, and step through the tests. Right: programs left untested after each test, against the straight line of testing them one by one.`}
    >
      <Plots cols={2} widths={[62, 38]}>
        <Plot x={xa} y={ya}>
          {view.groups.open.length > 0 && <Points name="open" {...view.pts(view.groups.open)} muted thin size={4} />}
          {(['generalisation', 'specialisation', 'elimination'] as const).map((c) =>
            view.groups[c].length > 0 ? (
              <Points key={c} name={`pruned: ${c}`} {...view.pts(view.groups[c])} slot={SLOT[c]} size={5} />
            ) : null,
          )}
          {view.groups.tested.length > 0 && (
            <Points name="tested earlier" {...view.pts(view.groups.tested)} emphasis size={6} />
          )}
          <Points
            name={step.solution ? 'solution' : 'tested now'}
            x={[view.current[0]]}
            y={[view.current[1]]}
            {...(step.solution ? { tone: 'success' as const } : { emphasis: true })}
            size={13}
            shape={3}
          />
        </Plot>
        <Plot x={xb} y={yb}>
          <Curve
            name="generate and test"
            x={[0, run.enumerateTests]}
            y={[run.programs.length, run.programs.length - run.enumerateTests]}
            muted
            dashed
          />
          <Curve
            name="with constraints"
            x={[0, ...tests.map((_, k) => k + 1)]}
            y={[run.programs.length, ...tests.map((s) => s.remaining)]}
            emphasis
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
