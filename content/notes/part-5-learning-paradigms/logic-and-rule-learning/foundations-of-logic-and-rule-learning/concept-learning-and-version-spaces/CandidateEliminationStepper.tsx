import { useState } from 'react'
import { trace } from 'aifn-compute/foundation/trace'
import {
  choice,
  cn,
  Figure,
  Player,
  Plot,
  Points,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  allInstances,
  conjunctionText,
  instanceText,
  type AttributeSpace,
  type AttributeInstance,
  type LabelledInstance,
  candidateElimination,
  versionSpaceMembers,
  versionSpaceVote,
  type CandidateEliminationState,
} from 'aifn-compute/logic/induction'

/** Four attributes of a day; the concept is "a good day for a picnic". */
const SPACE: AttributeSpace = {
  names: ['sky', 'temp', 'humidity', 'wind'],
  values: [
    ['sunny', 'cloudy', 'rainy'],
    ['warm', 'cold'],
    ['normal', 'high'],
    ['weak', 'strong'],
  ],
}

const CLEAN: LabelledInstance[] = [
  { x: [0, 0, 0, 0], positive: true },
  { x: [0, 0, 1, 0], positive: true },
  { x: [2, 1, 1, 1], positive: false },
  { x: [0, 1, 1, 0], positive: true },
  { x: [1, 0, 0, 0], positive: false },
  { x: [0, 0, 0, 1], positive: false },
]

const PRESETS: Record<string, { label: string; examples: LabelledInstance[] }> = {
  clean: { label: 'six noise-free examples', examples: CLEAN },
  negativesFirst: { label: 'the same six, negatives first', examples: [2, 4, 5, 0, 1, 3].map((i) => CLEAN[i]) },
  mislabelled: {
    label: 'a repeated day with the opposite label',
    examples: [...CLEAN, { x: [0, 1, 1, 0], positive: false }],
  },
  disjunctive: {
    label: 'target: sky = sunny or wind = weak',
    examples: [
      { x: [0, 0, 0, 1], positive: true },
      { x: [2, 0, 0, 0], positive: true },
      { x: [1, 0, 0, 1], positive: false },
    ],
  },
}

const COLUMNS = ['sunny warm', 'sunny cold', 'cloudy warm', 'cloudy cold', 'rainy warm', 'rainy cold']
const ROWS = ['normal weak', 'normal strong', 'high weak', 'high strong']
const cell = (x: AttributeInstance) => ({ col: x[0] * 2 + x[1], row: x[2] * 2 + x[3] })
const INSTANCES = allInstances(SPACE)

// Each preset's trace, and the version space and its vote on the 24 days at each step, are computed on first use and
// kept (at most 8 steps per preset, 109 hypotheses, 24 days).
type StepView = {
  members: ReturnType<typeof versionSpaceMembers>
  votes: ({ x: AttributeInstance } & ReturnType<typeof versionSpaceVote>)[]
}
const TRACES = new Map<string, CandidateEliminationState[]>()
const VIEWS = new Map<string, StepView>()
function traceOf(preset: string): CandidateEliminationState[] {
  let states = TRACES.get(preset)
  if (!states) {
    const examples = PRESETS[preset].examples
    states = trace(candidateElimination(SPACE, examples), undefined, examples.length + 1, { keep: 'all' })
      .steps as CandidateEliminationState[]
    TRACES.set(preset, states)
  }
  return states
}
function viewOf(preset: string, t: number, s: CandidateEliminationState): StepView {
  const key = `${preset}/${t}`
  let view = VIEWS.get(key)
  if (!view) {
    const members = versionSpaceMembers(SPACE, s.S, s.G)
    view = { members, votes: INSTANCES.map((x) => ({ x, ...versionSpaceVote(members, x) })) }
    VIEWS.set(key, view)
  }
  return view
}

const schema = {
  preset: choice(
    Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })),
    'clean',
    { label: 'training sequence' },
  ),
}

/** Candidate elimination, one example per step: the S and G boundaries and how the version space labels every day. */
export function CandidateEliminationStepper() {
  const state = useFigureState(schema)
  const examples = PRESETS[state.preset].examples
  const [pos, setPos] = useState({ preset: state.preset as string, step: 0 })
  const states = traceOf(state.preset)
  const last = states.length - 1
  const t = pos.preset === state.preset ? Math.min(pos.step, last) : 0
  const s = states[t]
  const { members, votes } = viewOf(state.preset, t, s)

  const xAxis = useAxis({ label: 'sky and temperature', categories: COLUMNS })
  const yAxis = useAxis({ label: 'humidity and wind', categories: ROWS })
  const layer = (verdict: string) => {
    const pts = votes.filter((v) => v.verdict === verdict).map((v) => cell(v.x))
    return { x: pts.map((p) => p.col), y: pts.map((p) => p.row) }
  }
  const positive = layer('positive')
  const negative = layer('negative')
  const disagree = layer('disagree')
  const seen = examples.slice(0, s.seen).map((e, i) => ({ ...cell(e.x), label: `${i + 1}${e.positive ? '+' : '−'}` }))

  const event = s.event
  const what =
    event.kind === 'start'
      ? 'Start: S holds only the empty hypothesis ∅, which covers nothing, and G holds ⟨?, ?, ?, ?⟩, which covers everything. Every hypothesis lies between them.'
      : event.kind === 'positive'
        ? `Example ${event.example + 1} is positive. ${event.droppedFromG.length ? `G loses ${event.droppedFromG.map((h) => conjunctionText(h, SPACE)).join(', ')}, which exclude it. ` : 'Every member of G covers it. '}${event.generalised.length ? event.generalised.map((g) => `S member ${conjunctionText(g.from, SPACE)} is generalised just enough to cover it: ${g.to.length ? g.to.map((h) => conjunctionText(h, SPACE)).join(', ') : 'no generalisation stays below G, so it is dropped'}.`).join(' ') : 'S already covers it.'}`
        : `Example ${event.example + 1} is negative. ${event.droppedFromS.length ? `S loses ${event.droppedFromS.map((h) => conjunctionText(h, SPACE)).join(', ')}, which covers it. ` : 'No member of S covers it. '}${event.specialised.length ? event.specialised.map((g) => `G member ${conjunctionText(g.from, SPACE)} covers it and is replaced by its minimal specialisations that exclude it and stay above S: ${g.to.length ? g.to.map((h) => conjunctionText(h, SPACE)).join(', ') : 'none'}.`).join(' ') : 'No member of G covers it.'}`

  return (
    <Figure
      title="Candidate elimination on the picnic table"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="examples used"
          value={t}
          onChange={(k) => setPos({ preset: state.preset, step: k })}
          count={states.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={
        <div className="text-left font-prose text-[15px] leading-snug">
          {what}
          {s.collapsed ? ' A boundary is empty: no conjunction is consistent with the examples.' : ''}
        </div>
      }
      readouts={
        <>
          <Readout
            label="S"
            value={s.S.length ? s.S.map((h) => conjunctionText(h, SPACE)).join('  ') : '∅ (empty set)'}
          />
          <Readout
            label="G"
            value={s.G.length ? s.G.map((h) => conjunctionText(h, SPACE)).join('  ') : '∅ (empty set)'}
          />
          <Readout label="hypotheses in the version space" value={`${members.length} of 109`} />
          <Readout
            label="days labelled unanimously"
            value={`${positive.x.length} positive, ${negative.x.length} negative, ${disagree.x.length} in dispute`}
          />
        </>
      }
      caption="Pick a training sequence and step through it. Each step uses one example. The grid shows all 24 possible days, coloured by how the hypotheses still in the version space label them: every one says positive, every one says negative, or they disagree. Numbered marks are the examples used so far. The version space contains exactly the conjunctions above some member of S and below some member of G."
    >
      <div className="grid h-full gap-4 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="h-72 md:h-full">
          <Plot x={xAxis} y={yAxis}>
            {positive.x.length > 0 && (
              <Points name="all say positive" x={positive.x} y={positive.y} slot={0} size={18} />
            )}
            {negative.x.length > 0 && (
              <Points name="all say negative" x={negative.x} y={negative.y} slot={1} size={18} />
            )}
            {disagree.x.length > 0 && <Points name="they disagree" x={disagree.x} y={disagree.y} slot={2} size={18} />}
            {seen.length > 0 && (
              <Points
                name="examples used"
                x={seen.map((p) => p.col)}
                y={seen.map((p) => p.row)}
                labels={seen.map((p) => p.label)}
                emphasis
                size={7}
              />
            )}
          </Plot>
        </div>
        <div className="overflow-x-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>#</TableHead>
                <TableHead>day ({SPACE.names.join(', ')})</TableHead>
                <TableHead>label</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {examples.map((e, i) => (
                <TableRow
                  key={i}
                  className={cn(i >= s.seen && 'text-muted-foreground', i === s.seen - 1 && 'bg-muted font-semibold')}
                >
                  <TableCell>{i + 1}</TableCell>
                  <TableCell className="font-mono">{instanceText(e.x, SPACE)}</TableCell>
                  <TableCell>{e.positive ? 'yes' : 'no'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {members.length > 0 && members.length <= 16 && (
            <div className="mt-2 text-xs">
              <div className="text-muted-foreground">the version space</div>
              <div className="font-mono">{members.map((h) => conjunctionText(h, SPACE)).join('  ')}</div>
            </div>
          )}
        </div>
      </div>
    </Figure>
  )
}
