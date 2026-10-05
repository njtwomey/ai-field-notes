/**
 * Showcase: FOIL. Background facts (a family tree), positive and negative examples of a target relation (clickable
 * cells), and aifn's `foilSteps` traced step by step: the clause growing literal by literal, every candidate literal
 * with its FOIL gain, the examples the clause covers, and the learned program, which the Prolog engine of
 * `aifn/logic/resolution` then runs.
 */
import { useMemo, useState } from 'react'
import { fromEdges } from 'aifn-compute/graph'
import { trace } from 'aifn-compute/foundation/trace'
import { foilProblem, foilSteps, type FoilCandidate, type FoilState } from 'aifn-compute/logic/induction'
import { solveQuery } from 'aifn-compute/logic/resolution'
import { Button, Player, Select, StatusText } from 'aifn-render/controls'
import { seriesColor, useTheme } from 'aifn-render/design'
import { Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { Input } from 'aifn-render/ui/input'
import { cn } from 'aifn-render/lib/utils'
import { GraphView } from '@lab/views'
import { Readout } from 'aifn-render/viz'

type Task = {
  label: string
  target: string
  people: readonly string[]
  female?: readonly string[]
  parents: readonly (readonly [string, string])[]
  /** The relation to learn, as Prolog over `parent/2` (and `female/1`), used for the default examples. */
  truth: string
  queries: readonly string[]
  note: string
}

const SMALL: Task['parents'] = [
  ['ann', 'mary'],
  ['ann', 'tom'],
  ['tom', 'eve'],
  ['tom', 'ian'],
]
const LARGE: Task['parents'] = [
  ['george', 'bob'],
  ['george', 'liz'],
  ['mum', 'bob'],
  ['mum', 'liz'],
  ['bob', 'ann'],
  ['bob', 'pat'],
  ['liz', 'kim'],
  ['pat', 'jim'],
  ['pat', 'sue'],
  ['kim', 'tim'],
]
const LARGE_PEOPLE = ['george', 'mum', 'bob', 'liz', 'ann', 'pat', 'kim', 'jim', 'sue', 'tim']

const TASKS = {
  daughter: {
    label: 'daughter (Lavrač and Džeroski)',
    target: 'daughter',
    people: ['ann', 'mary', 'tom', 'eve', 'ian'],
    female: ['ann', 'mary', 'eve'],
    parents: SMALL,
    truth: 'target(X, Y) :- female(X), parent(Y, X).',
    queries: ['daughter(D, tom)', 'daughter(mary, P)'],
    note: 'daughter(X, Y): X is a daughter of Y. Background: parent/2 and female/1.',
  },
  grandparent: {
    label: 'grandparent',
    target: 'grandparent',
    people: LARGE_PEOPLE,
    parents: LARGE,
    truth: 'target(X, Z) :- parent(X, Y), parent(Y, Z).',
    queries: ['grandparent(george, G)', 'grandparent(G, jim)'],
    note: 'grandparent(X, Z): X is a grandparent of Z. Background: parent/2. FOIL must invent the middle person.',
  },
  ancestor: {
    label: 'ancestor (recursive)',
    target: 'ancestor',
    people: LARGE_PEOPLE,
    parents: LARGE,
    truth: 'target(X, Y) :- parent(X, Y).\ntarget(X, Y) :- parent(X, Z), target(Z, Y).',
    queries: ['ancestor(george, D)', 'ancestor(A, tim)'],
    note: 'ancestor(X, Y): a recursive relation. FOIL may use ancestor in a body, read off the positive examples.',
  },
} satisfies Record<string, Task>
type TaskName = keyof typeof TASKS

type Label = '+' | '-' | '·'
const NEXT: Record<Label, Label> = { '+': '-', '-': '·', '·': '+' }

const backgroundOf = (task: Task) =>
  [...task.parents.map(([p, c]) => `parent(${p}, ${c}).`), ...(task.female ?? []).map((f) => `female(${f}).`)].join(
    '\n',
  )

/** The default labels: the true relation positive, every other pair negative (the closed world). */
function defaultLabels(task: Task): Record<string, Label> {
  const truth = new Set(
    solveQuery(`${backgroundOf(task)}\n${task.truth}`, 'target(A, B)').solutions.map((s) =>
      s.bindings.map((b) => (b.value.kind === 'atom' ? b.value.name : '')).join(','),
    ),
  )
  const out: Record<string, Label> = {}
  for (const x of task.people) for (const y of task.people) out[`${x},${y}`] = truth.has(`${x},${y}`) ? '+' : '-'
  return out
}

const fmtGain = (g: number) => (Number.isFinite(g) ? g.toFixed(2) : '−∞')

function explain(s: FoilState): string {
  const e = s.event
  switch (e.kind) {
    case 'start':
      return `Step 0: the first clause is ${s.currentText} with an empty body: it covers every positive and every negative example. FOIL now adds literals, one per step, until no negative is covered.`
    case 'literal':
      return `Step ${s.t}: add ${e.chosen.text}, the literal of highest FOIL gain (${fmtGain(e.chosen.gain)}): the clause now covers ${s.coveredPositives.length} positive and ${s.coveredNegatives.length} negative examples${s.coveredNegatives.length === 0 ? ', none negative, so it is finished' : ''}.`
    case 'clause':
      return `Step ${s.t}: learned ${e.clause.text}${e.clause.removed.length ? ` (simplified from ${e.clause.grown}: ${e.clause.removed.join(', ')} was not needed)` : ''}. It covers ${e.clause.newlyCovered.length} more positive example${e.clause.newlyCovered.length === 1 ? '' : 's'}; ${s.uncovered.length ? `${s.uncovered.length} remain, so FOIL starts a new clause.` : 'none remain: the definition is complete.'}`
    case 'stuck':
      return `Step ${s.t}: FOIL stops: ${e.reason}.`
    case 'done':
      return `Step ${s.t}: done.`
  }
}

export function FoilShowcase() {
  const { resolved: mode } = useTheme()
  const [taskName, setTaskName] = useState<TaskName>('daughter')
  const task: Task = TASKS[taskName]
  const [labels, setLabels] = useState<Record<string, Label>>(() => defaultLabels(TASKS.daughter))
  const [step, setStep] = useState(0)
  const [hoverCandidate, setHoverCandidate] = useState<FoilCandidate | null>(null)
  const [hoverPair, setHoverPair] = useState<[string, string] | null>(null)
  const [query, setQuery] = useState<string>(TASKS.daughter.queries[0])

  const pickTask = (name: TaskName) => {
    setTaskName(name)
    setLabels(defaultLabels(TASKS[name]))
    setStep(0)
    setQuery(TASKS[name].queries[0])
  }

  const background = useMemo(() => backgroundOf(task), [task])
  const learned = useMemo(() => {
    const pairs = Object.entries(labels)
    const pos = pairs.filter(([, l]) => l === '+').map(([k]) => k)
    const neg = pairs.filter(([, l]) => l === '-').map(([k]) => k)
    if (pos.length === 0) return { ok: false as const, message: 'Mark at least one positive example.' }
    const atoms = (keys: string[]) => keys.map((k) => `${task.target}(${k}).`).join(' ')
    const problem = foilProblem(background, atoms(pos), atoms(neg))
    const run = trace(foilSteps(problem), undefined, 60, { keep: 'all' })
    const states = run.steps as FoilState[]
    const name = (e: readonly number[]) => e.map((c) => problem.constants[c]).join(',')
    return {
      ok: true as const,
      problem,
      states,
      posKeys: problem.positives.map(name),
      negKeys: problem.negatives.map(name),
    }
  }, [labels, background, task])

  const states = learned.ok ? learned.states : []
  const last = Math.max(0, states.length - 1)
  const t = Math.min(step, last)
  const s = states[t] as FoilState | undefined
  const final = states[last] as FoilState | undefined
  const program = (s?.clauses ?? []).map((c) => c.text).join('\n')
  const finalProgram = (final?.clauses ?? []).map((c) => c.text).join('\n')

  // Which examples to mark as covered: a hovered candidate's, else the clause being grown.
  const covered = useMemo(() => {
    if (!learned.ok || !s) return new Set<string>()
    const pos = hoverCandidate?.positives ?? s.coveredPositives
    const neg = hoverCandidate?.negatives ?? s.coveredNegatives
    return new Set([...pos.map((i) => learned.posKeys[i]), ...neg.map((i) => learned.negKeys[i])])
  }, [learned, s, hoverCandidate])
  const done = useMemo(() => {
    if (!learned.ok || !s) return new Set<string>()
    const remaining = new Set(s.uncovered)
    return new Set(learned.posKeys.filter((_, i) => !remaining.has(i)))
  }, [learned, s])

  const graph = useMemo(() => {
    const index = new Map(task.people.map((p, i) => [p, i]))
    return fromEdges(
      task.people.length,
      task.parents.map(([p, c]) => [index.get(p)!, index.get(c)!] as [number, number]),
    )
  }, [task])

  const answers = useMemo(() => {
    if (!finalProgram) return { message: 'Nothing learned yet.', answers: [] as readonly string[] }
    try {
      const r = solveQuery(`${background}\n${finalProgram}`, query, { maxSteps: 20_000 })
      return { message: r.message, answers: r.answers }
    } catch (e) {
      return { message: (e as Error).message, answers: [] as readonly string[] }
    }
  }, [background, finalProgram, query])

  const posColor = seriesColor(mode, 2)
  const negColor = seriesColor(mode, 7)
  const candidates = s?.candidates ?? []
  const chosen = s?.event.kind === 'literal' ? s.event.chosen.text : null

  return (
    <Figure
      title="Learning rules from examples: FOIL"
      purpose="FOIL learns a Prolog definition of a relation from facts and examples: it grows one clause at a time, adding the literal with the highest information gain until the clause covers no negative example, then removes the positives it covers and starts the next clause."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <Select
            label="task"
            value={taskName}
            onChange={pickTask}
            options={(Object.keys(TASKS) as TaskName[]).map((k) => ({ value: k, label: TASKS[k].label }))}
          />
          <div className="flex items-end">
            <Button size="sm" variant="outline" onClick={() => setLabels(defaultLabels(task))}>
              Reset examples
            </Button>
          </div>
          <Player
            label="FOIL step"
            value={t}
            onChange={setStep}
            count={states.length || 1}
            format={(p) => `${p} of ${last}`}
          />
        </>
      }
      equation={
        <div className="flex flex-col gap-2 text-left">
          <div className="font-prose text-[15px] leading-snug">
            {learned.ok && s ? explain(s) : learned.ok ? '' : learned.message}
          </div>
          <div className="font-mono text-sm">
            <span className="text-muted-foreground">growing: </span>
            {s?.terminated ? <span className="text-muted-foreground">(finished)</span> : s?.currentText}
          </div>
          <div className="font-mono text-sm">
            <span className="text-muted-foreground">learned so far: </span>
            {program ? (
              <pre className="mt-0.5 whitespace-pre-wrap">{program}</pre>
            ) : (
              <span className="text-muted-foreground">nothing yet</span>
            )}
          </div>
        </div>
      }
      readouts={
        s && {
          [`step ${t}`]: (
            <>
              <Readout label="positive tuples p" value={s.current.pos.length} />
              <Readout label="negative tuples n" value={s.current.neg.length} />
              <Readout
                label="examples covered (+ / −)"
                value={`${s.coveredPositives.length} / ${s.coveredNegatives.length}`}
              />
              <Readout label="positives left" value={s.uncovered.length} />
              <Readout label="clauses" value={s.clauses.length} />
            </>
          ),
        }
      }
      caption={
        <>
          {task.note} Each cell of the grid is a pair (row X, column Y) of the target relation: + a positive example, −
          a negative, · unused. Click a cell to cycle it (+ → − → · → +); by default the true relation is positive and
          every other pair negative (the closed world). Play or step FOIL from step 0. Filled cells are the examples the
          clause being grown covers; ✓ marks positives covered by clauses already learned. The table lists every literal
          FOIL could add at this step with p and n (positive and negative tuples, i.e. bindings of the clause&apos;s
          variables, after adding it), t (positive tuples before that survive) and the gain t · (log₂ p₁/(p₁+n₁) − log₂
          p₀/(p₀+n₀)); the chosen one is bold, and hovering a row shows what it would cover. New variables (C, D, …)
          range over the people in the facts. The learned program is run by the Prolog engine of the Prolog showcase on
          the query at the bottom right.
        </>
      }
    >
      <Dashboard>
        <DashboardRow ratio={1.5}>
          <DashboardCell ratio={1}>
            <div className="flex h-full flex-col gap-1 overflow-auto text-xs">
              <div className="text-muted-foreground">examples of {task.target}(X, Y): rows X, columns Y</div>
              <table className="w-auto border-collapse self-start font-mono">
                <thead>
                  <tr>
                    <th />
                    {task.people.map((y) => (
                      <th key={y} className="px-0.5 pb-1 text-[10px] font-normal text-muted-foreground">
                        <div className="w-6 truncate" title={y}>
                          {y}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {task.people.map((x) => (
                    <tr key={x}>
                      <th className="pr-1 text-right text-[10px] font-normal text-muted-foreground">{x}</th>
                      {task.people.map((y) => {
                        const k = `${x},${y}`
                        const l = labels[k] ?? '·'
                        const color = l === '+' ? posColor : l === '-' ? negColor : undefined
                        const on = covered.has(k)
                        return (
                          <td key={y} className="p-px">
                            <button
                              type="button"
                              aria-label={`${task.target}(${x}, ${y}): ${l === '+' ? 'positive' : l === '-' ? 'negative' : 'unused'}`}
                              className="flex size-6 items-center justify-center rounded border text-[11px]"
                              style={{
                                borderColor: color ?? 'var(--border)',
                                background: on && color ? color : undefined,
                                color: on ? 'var(--background)' : color,
                              }}
                              onClick={() => {
                                setLabels({ ...labels, [k]: NEXT[l] })
                                setStep(0)
                              }}
                              onMouseEnter={() => setHoverPair([x, y])}
                              onMouseLeave={() => setHoverPair(null)}
                            >
                              {done.has(k) && !on ? '✓' : l === '-' ? '−' : l}
                            </button>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </DashboardCell>
          <DashboardCell ratio={1.1}>
            <GraphView
              graph={graph}
              ariaLabel="The family tree: parent → child"
              height="fill"
              nodeLabels={task.people.map((p) => p)}
              nodeTone={(v) => (task.female?.includes(task.people[v]) ? 4 : 'ink')}
              nodeHighlight={(v) => hoverPair !== null && hoverPair.includes(task.people[v])}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={1}>
          <DashboardCell ratio={1.3}>
            <div className="h-full overflow-auto text-xs" onMouseLeave={() => setHoverCandidate(null)}>
              <div className="mb-1 text-muted-foreground">
                {candidates.length
                  ? `candidate literals at step ${t} (${candidates.length}), best first`
                  : 'no candidates scored at this step'}
              </div>
              {candidates.length > 0 && (
                <table className="w-full border-collapse tabular-nums">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="px-2 py-0.5 font-normal">literal</th>
                      <th className="px-2 py-0.5 font-normal">p</th>
                      <th className="px-2 py-0.5 font-normal">n</th>
                      <th className="px-2 py-0.5 font-normal">t</th>
                      <th className="px-2 py-0.5 font-normal">gain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {candidates.slice(0, 40).map((c) => (
                      <tr
                        key={c.text}
                        className={cn('border-b border-border/40 hover:bg-muted', c.text === chosen && 'font-semibold')}
                        onMouseEnter={() => setHoverCandidate(c)}
                      >
                        <td className="px-2 py-0.5 font-mono">{c.text}</td>
                        <td className="px-2 py-0.5">{c.p}</td>
                        <td className="px-2 py-0.5">{c.n}</td>
                        <td className="px-2 py-0.5">{c.t}</td>
                        <td className="px-2 py-0.5">{fmtGain(c.gain)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </DashboardCell>
          <DashboardCell ratio={1}>
            <div className="flex h-full flex-col gap-2 overflow-auto text-xs">
              <div className="text-muted-foreground">
                the learned program (after the last step), run by the Prolog engine
              </div>
              <pre className="font-mono text-sm whitespace-pre-wrap">{finalProgram || '(nothing learned)'}</pre>
              <label className="flex flex-col gap-1">
                <span className="text-muted-foreground">query (?-)</span>
                <Input
                  aria-label="query for the learned program"
                  className="h-8 font-mono text-sm"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap gap-1.5">
                {task.queries.map((q) => (
                  <Button
                    key={q}
                    size="sm"
                    variant="outline"
                    className="h-7 font-mono text-xs"
                    onClick={() => setQuery(q)}
                  >
                    ?- {q}
                  </Button>
                ))}
              </div>
              <StatusText>{answers.message}</StatusText>
              <ol className="font-mono">
                {answers.answers.slice(0, 20).map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ol>
            </div>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
