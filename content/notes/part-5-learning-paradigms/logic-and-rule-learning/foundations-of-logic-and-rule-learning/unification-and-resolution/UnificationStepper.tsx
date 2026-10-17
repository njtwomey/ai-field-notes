import { useMemo, useState } from 'react'
import {
  applySubstitution,
  parseQuery,
  substitutionToString,
  termToString,
  unificationTrace,
  type UnificationRule,
} from 'aifn-compute/logic/terms'
import { choice, cn, Figure, Player, Readout, setting, useFigureState } from 'aifn-render'

const PAIRS = [
  { value: 'worked', label: 'p(f(X), Y, b) = p(Z, g(Z), W)' },
  { value: 'chain', label: 'f(X, Y, Z) = f(g(Y), h(Z), a)' },
  { value: 'clash', label: 'f(X, X) = f(a, b)' },
  { value: 'occurs', label: 'f(X, g(X)) = f(Y, Y)' },
  { value: 'head', label: 'grandparent(ann, W) = grandparent(X1, Z1)' },
  { value: 'list', label: 'p(X, [Y | T]) = p(1, [2, 3])' },
] as const

const RULE_TEXT: Record<UnificationRule, string> = {
  start: 'start',
  delete: 'delete',
  decompose: 'decompose',
  conflict: 'conflict: fail',
  swap: 'swap',
  eliminate: 'eliminate',
  occurs: 'occurs check: fail',
  done: 'done',
}

const schema = {
  pair: choice(PAIRS, 'worked', { label: 'equation s = t' }),
  occursCheck: setting(true, 'occurs check'),
}

/** Martelli and Montanari's unification, one rule per step. */
export function UnificationStepper() {
  const state = useFigureState(schema)
  // The step resets to 0 when the equation or the occurs check changes.
  const key = `${state.pair}/${state.occursCheck}`
  const [pos, setPos] = useState({ key, step: 0 })
  const setStep = (k: number) => setPos({ key, step: k })
  const { steps, a } = useMemo(() => {
    const text = PAIRS.find((p) => p.value === state.pair)!.label.replace(' = ', ', ')
    const { goals } = parseQuery(text)
    return { steps: unificationTrace(goals[0], goals[1], { occursCheck: state.occursCheck }), a: goals[0] }
  }, [state.pair, state.occursCheck])
  const t = pos.key === key ? Math.min(pos.step, steps.length - 1) : 0
  const s = steps[t]
  const last = steps[steps.length - 1]

  return (
    <Figure
      title="Unification as solving equations"
      state={state}
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={setStep}
          count={steps.length}
          format={(k) => `${k} of ${steps.length - 1}`}
        />
      }
      readouts={
        <>
          <Readout label="rule" value={RULE_TEXT[s.rule]} />
          <Readout label="equations left" value={s.equations.length} />
          <Readout label="bindings" value={s.solved.length} />
          <Readout
            label="result"
            value={
              last.status === 'failed'
                ? 'no unifier'
                : `mgu ${substitutionToString(last.solved)}; common instance ${termToString(applySubstitution(a, last.solved))}`
            }
          />
        </>
      }
      caption="Pick an equation between two terms and step through it. Each step treats the first equation still to solve: delete it when both sides are identical, decompose it when both sides start with the same function symbol, fail when they start with different ones, swap it to put a variable on the left, and eliminate a variable by binding it and substituting it everywhere else. The occurs check makes X = f(X) fail; switched off, X is bound to a term containing itself, as in Prolog's =/2."
    >
      <div className="grid gap-4 text-sm md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs text-muted-foreground">equations still to solve (first is next)</div>
          {s.equations.length === 0 ? (
            <div className="text-muted-foreground">none</div>
          ) : (
            <ol className="flex flex-col gap-1 font-mono">
              {s.equations.map(([l, r], i) => (
                <li key={i} className={cn(i === 0 && s.status === 'running' && 'font-semibold')}>
                  {termToString(l)} = {termToString(r)}
                </li>
              ))}
            </ol>
          )}
          <div className="mt-3 mb-1 text-xs text-muted-foreground">solved bindings</div>
          <div className="font-mono">{s.solved.length ? substitutionToString(s.solved) : '{}'}</div>
        </div>
        <div className={cn('font-prose text-[15px] leading-snug', s.status === 'failed' && 'font-semibold')}>
          <div className="mb-1 font-sans text-xs text-muted-foreground">step {t}</div>
          {s.note}
        </div>
      </div>
    </Figure>
  )
}
