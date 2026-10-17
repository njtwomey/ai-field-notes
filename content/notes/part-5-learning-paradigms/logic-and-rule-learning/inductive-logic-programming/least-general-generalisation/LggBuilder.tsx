import { useMemo, useState } from 'react'
import {
  reduceClause,
  clauseLggTrace,
  relativeLgg,
  termLggTraced,
  type LggPair,
  type LggStep,
  type LiteralPairStep,
} from 'aifn-compute/logic/induction'
import { clauseToString, parseProgram, parseTerm, termToString, type Term } from 'aifn-compute/logic/terms'
import {
  choice,
  cn,
  Figure,
  Player,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
} from 'aifn-render'

/** The small family of the coverage figure in the first-order logic note. */
const SMALL_FAMILY =
  'parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian). female(ann). female(mary). female(eve).'

const PRESETS = [
  { value: 'nested', label: 'terms f(a, g(a)) and f(b, g(b))', kind: 'terms', a: 'f(a, g(a))', b: 'f(b, g(b))' },
  { value: 'lists', label: 'lists [1, 2, 3] and [4, 2, 5]', kind: 'terms', a: '[1, 2, 3]', b: '[4, 2, 5]' },
  { value: 'swap', label: 'terms f(a, b) and f(b, a)', kind: 'terms', a: 'f(a, b)', b: 'f(b, a)' },
  {
    value: 'clauses',
    label: 'two daughter clauses',
    kind: 'clauses',
    a: 'daughter(mary, ann) :- female(mary), female(ann), parent(ann, mary).',
    b: 'daughter(eve, tom) :- female(eve), parent(tom, eve).',
  },
  {
    value: 'relative',
    label: 'relative lgg of two daughter examples (Golem)',
    kind: 'relative',
    a: 'daughter(mary, ann)',
    b: 'daughter(eve, tom)',
  },
] as const

const schema = {
  preset: choice(
    PRESETS.map(({ value, label }) => ({ value, label })),
    'nested',
    { label: 'inputs' },
  ),
}

const show = (t: Term) => termToString(t)
const ACTION: Record<LggStep['action'], string> = {
  same: 'identical: kept',
  descend: 'same functor: compare the arguments',
  new: 'differ: a new variable',
  reuse: 'differ, pair seen before: same variable',
}

/** One frame of the walk-through: the lines built so far, the table of pairs, and a sentence. */
interface Frame {
  readonly lines: readonly { text: string; struck?: boolean; current?: boolean }[]
  readonly pairs: readonly LggPair[]
  readonly note: string
}

function termFrames(a: Term, b: Term): { frames: Frame[]; result: string } {
  const steps: LggStep[] = []
  const table = { pairs: [] as LggPair[] }
  const result = termLggTraced(a, b, table, steps)
  const frames: Frame[] = [{ lines: [], pairs: [], note: `Compare ${show(a)} with ${show(b)} from the root down.` }]
  steps.forEach((s, k) => {
    const pairs = table.pairs.filter((p) =>
      steps.slice(0, k + 1).some((x) => x.action === 'new' && x.variable === p.variable),
    )
    frames.push({
      lines: steps.slice(0, k + 1).map((x, i) => ({
        text: `${x.path.length ? `argument ${x.path.map((p) => p + 1).join('.')}` : 'root'}: ${show(x.left)} vs ${show(x.right)}: ${ACTION[x.action]}${x.variable ? ` ${show(x.variable)}` : ''}`,
        current: i === k,
      })),
      pairs,
      note:
        s.action === 'new'
          ? `${show(s.left)} and ${show(s.right)} differ, and the pair is new, so it becomes the variable ${show(s.variable!)}.`
          : s.action === 'reuse'
            ? `${show(s.left)} and ${show(s.right)} form a pair already in the table, so they become ${show(s.variable!)} again. This is what keeps the generalisation specific.`
            : s.action === 'same'
              ? `${show(s.left)} occurs in both terms at this position, so it is kept.`
              : `Both are ${s.left.kind === 'compound' ? s.left.functor : ''} with the same number of arguments, so the arguments are compared in turn.`,
    })
  })
  frames.push({ ...frames[frames.length - 1], note: `The least general generalisation is ${show(result)}.` })
  return { frames, result: show(result) }
}

function clauseFrames(steps: readonly LiteralPairStep[], left: readonly Term[], right: readonly Term[]): Frame[] {
  const frames: Frame[] = [
    {
      lines: [],
      pairs: [],
      note: 'Pair the heads, then every two body literals with the same predicate, sharing one table.',
    },
  ]
  steps.forEach((s, k) => {
    frames.push({
      lines: steps.slice(0, k + 1).map((x, i) => ({
        text: `${i === 0 ? 'head' : 'body'}: ${show(left[x.left])} with ${show(right[x.right])} gives ${show(x.literal)}`,
        struck: x.duplicate,
        current: i === k,
      })),
      pairs: s.pairs,
      note: s.duplicate
        ? `${show(s.literal)} repeats a literal already in the body and is dropped.`
        : `${show(left[s.left])} and ${show(right[s.right])} generalise to ${show(s.literal)}.`,
    })
  })
  return frames
}

/** The least general generalisation built one comparison (terms) or one literal pair (clauses) at a time. */
export function LggBuilder() {
  const state = useFigureState(schema)
  const preset = PRESETS.find((p) => p.value === state.preset) ?? PRESETS[0]
  const [pos, setPos] = useState({ preset: preset.value as string, step: 0 })

  const run = useMemo(() => {
    if (preset.kind === 'terms') {
      const { frames, result } = termFrames(parseTerm(preset.a), parseTerm(preset.b))
      return { frames, result, sizes: null as string | null }
    }
    if (preset.kind === 'clauses') {
      const c = parseProgram(preset.a).clauses[0]
      const d = parseProgram(preset.b).clauses[0]
      const out = clauseLggTrace(c, d)!
      const frames = clauseFrames(out.steps, [c.head, ...c.body], [d.head, ...d.body])
      const reduced = reduceClause(out.clause)
      frames.push({
        ...frames[frames.length - 1],
        note: `The lgg is ${clauseToString(out.clause)} Plotkin's reduction removes the literals another literal can be folded onto: ${clauseToString(reduced)}`,
      })
      return {
        frames,
        result: clauseToString(reduced),
        sizes: `${out.clause.body.length} body literals, ${reduced.body.length} after reduction`,
      }
    }
    const background = parseProgram(SMALL_FAMILY).clauses.map((c) => c.head)
    const e1 = parseTerm(preset.a)
    const e2 = parseTerm(preset.b)
    const r = relativeLgg(e1, e2, background)!
    const lits = [e1, ...background]
    const frames = clauseFrames(r.steps, lits, [e2, ...background])
    frames[0] = {
      ...frames[0],
      note: `Build the clauses ${show(e1)} ← B and ${show(e2)} ← B, with B the 7 background facts, and take their lgg.`,
    }
    frames.push({
      ...frames[frames.length - 1],
      note: `The raw relative lgg has ${r.raw.body.length} body literals. ${r.ground.length} are ground facts of B, and ${r.unlinked.length} share no variable with the head, even through other literals. B makes both kinds true whatever the head's values, so they are dropped, leaving ${r.linked.body.length}: ${clauseToString(r.linked)}`,
    })
    frames.push({
      ...frames[frames.length - 1],
      note: `Plotkin's reduction removes the literals another literal can be folded onto: the parent literals with an unused child variable fold onto parent(B, A). The result is ${clauseToString(r.clause)}`,
    })
    return {
      frames,
      result: clauseToString(r.clause),
      sizes: `${r.raw.body.length} raw, ${r.linked.body.length} linked to the head, ${r.clause.body.length} after reduction`,
    }
  }, [preset])

  const last = run.frames.length - 1
  const t = pos.preset === preset.value ? Math.min(pos.step, last) : 0
  const f = run.frames[t]
  const shown = f.lines.length > 14 ? f.lines.slice(f.lines.length - 14) : f.lines

  return (
    <Figure
      title="Building a least general generalisation"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={(k) => setPos({ preset: preset.value, step: k })}
          count={run.frames.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={<div className="text-left font-prose text-[15px] leading-snug">{f.note}</div>}
      readouts={
        <>
          <Readout label="inputs" value={`${preset.a}   |   ${preset.b}`} />
          <Readout label="result" value={t === last ? run.result : 'step to the end'} />
          {run.sizes && <Readout label="size" value={run.sizes} />}
        </>
      }
      caption="Pick two terms, two clauses or two examples and step through. For terms, each step compares one pair of subterms from the root down: identical subterms are kept, subterms with the same functor are compared argument by argument, and any other pair becomes a variable, the same pair always the same variable. For clauses, each step generalises the heads or one pair of body literals with the same predicate, sharing the table of pairs on the right. The relative preset takes the lgg of the two examples, each with the 7 background facts as its body, drops the literals the background makes true whatever the head's values, and reduces the rest."
    >
      <div className="grid h-full gap-3 overflow-auto md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <ol className="font-mono text-xs leading-relaxed">
          {f.lines.length > shown.length && (
            <li className="text-muted-foreground">… {f.lines.length - shown.length} earlier steps</li>
          )}
          {shown.map((l, i) => (
            <li key={i} className={cn(l.current && 'font-semibold', l.struck && 'text-muted-foreground line-through')}>
              {l.text}
            </li>
          ))}
        </ol>
        <Table className="text-xs">
          <TableHeader>
            <TableRow>
              <TableHead>first</TableHead>
              <TableHead>second</TableHead>
              <TableHead>variable</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {f.pairs.map((p) => (
              <TableRow key={show(p.variable)}>
                <TableCell className="font-mono">{show(p.left)}</TableCell>
                <TableCell className="font-mono">{show(p.right)}</TableCell>
                <TableCell className="font-mono">{show(p.variable)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
