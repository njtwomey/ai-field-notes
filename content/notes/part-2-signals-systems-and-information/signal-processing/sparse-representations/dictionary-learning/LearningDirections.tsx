import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { directionData, HIDDEN, learnDirections, worstAngle } from './directions'

const REACH = 1.45
const NAMES = ['atom 1', 'atom 2', 'atom 3', 'atom 4', 'atom 5']

export function LearningDirections() {
  const state = useFigureState({
    update: choice(
      [
        { value: 'ksvd', label: 'K-SVD' },
        { value: 'mod', label: 'method of optimal directions' },
      ],
      'ksvd',
      { label: 'dictionary update' },
    ),
    init: choice(
      [
        { value: 'signals', label: 'k training signals' },
        { value: 'directions', label: 'k random directions' },
      ],
      'signals',
      { label: 'start from' },
    ),
    k: int(3, { min: 2, max: 5, label: 'atoms k' }),
    noise: float(0.05, { min: 0, max: 0.3, step: 0.01, label: 'noise', suggestions: [0, 0.05, 0.15] }),
    start: int(1, { min: 1, max: 9999, label: 'start seed' }),
  })
  const { update, init, k, noise, start } = state

  const Y = useMemo(() => directionData(noise), [noise])
  const run = useComputed(() => learnDirections(Y, k, update, init, start), [Y, k, update, init, start])
  const rounds = run.value

  const key = `${update}|${init}|${k}|${noise}|${start}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = pos.key === key ? Math.min(pos.step, rounds.length - 1) : 0
  const round = rounds[step]

  const view = useMemo(() => {
    const { D, X } = round
    const n = Y[0].length
    const group: number[] = []
    const fit: { from: [number, number]; to: [number, number] }[] = []
    for (let i = 0; i < n; i++) {
      let j = X.findIndex((row) => row[i] !== 0)
      if (j < 0) j = 0
      group.push(j)
      const c = X[j][i]
      fit.push({ from: [Y[0][i], Y[1][i]], to: [c * D[0][j], c * D[1][j]] })
    }
    const lines = D[0].map((_, j) => ({
      from: [-REACH * D[0][j], -REACH * D[1][j]] as [number, number],
      to: [REACH * D[0][j], REACH * D[1][j]] as [number, number],
    }))
    const arrows = D[0].map((_, j) => ({
      from: [0, 0] as [number, number],
      to: [D[0][j], D[1][j]] as [number, number],
      slot: j,
      label: `d${j + 1}`,
    }))
    return { group, fit, lines, arrows }
  }, [round, Y])

  const hidden = useMemo(
    () =>
      HIDDEN.map((a) => ({
        from: [-REACH * Math.cos(a), -REACH * Math.sin(a)] as const,
        to: [REACH * Math.cos(a), REACH * Math.sin(a)] as const,
      })),
    [],
  )
  const curve = useMemo(
    () => ({ x: rounds.map((_, t) => t), y: rounds.map((r) => Math.max(r.objective, 1e-6)) }),
    [rounds],
  )

  const px = useAxis({ label: 'y₁', range: [-1.5, 1.5] })
  const py = useAxis({ label: 'y₂', range: [-1.5, 1.5], equal: px })
  const ox = useAxis({ label: 'round', integer: true, range: [0, undefined] })
  const oy = useAxis({ label: '½‖Y − DX‖²', log: true })

  return (
    <Figure
      title="Learning three directions from points"
      state={state}
      defaultSize="L"
      caption="Each point is a multiple of one of three hidden unit vectors (dashed) plus noise, so one atom per signal (s = 1) can represent it. Round 0 codes the points over the initial dictionary; each later round recodes every point by its best atom (the colour) and then moves the atoms. The thin grey segments join each point to its approximation x d_j; the objective is half the sum of their squared lengths. Like k-means, the fit alternates assignment and update, but an atom is a line through the origin, not a centre, and a point can sit on either side. Some starts end in a local minimum: with K-SVD, start seed 8 leaves two atoms on one line and one line without an atom; the method of optimal directions escapes it from the same start. With k = 2 one atom must serve two lines; with k = 4 or 5 the spare atoms split a line's points between them."
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={rounds.length}
          label="round"
          format={(t) => (t === 0 ? 'initial dictionary' : `round ${t} of ${rounds.length - 1}`)}
        />
      }
      readouts={
        <>
          <Readout label="objective ½‖Y − DX‖²" value={formatNumber(round.objective)} />
          <Readout label="largest angle to a hidden direction" value={`${worstAngle(round.D).toFixed(1)}°`} />
          <Readout label="atoms replaced this round" value={String(round.replaced)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={px} y={py} fitHeight>
          <Segments name="hidden directions" segments={hidden} muted dashed />
          <Segments name="error y − x d" segments={view.fit} muted width={0.75} />
          {view.lines.map((line, j) => (
            <Segments key={j} name={NAMES[j]} segments={[line]} slot={j} width={1} />
          ))}
          <Points name="signals" x={Y[0]} y={Y[1]} group={view.group} groupNames={NAMES.slice(0, k)} size={6} />
          <Vectors name="atoms" vectors={view.arrows} />
        </Plot>
        <Plot x={ox} y={oy} height={300}>
          <Curve name="objective" x={curve.x} y={curve.y} slot={0} showPoints />
          <Points name="this round" x={[step]} y={[curve.y[step]]} emphasis size={10} live />
        </Plot>
      </div>
    </Figure>
  )
}
