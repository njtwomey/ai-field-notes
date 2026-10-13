import { useMemo, useState } from 'react'
import { shatteringTable, type ShatterClass } from 'aifn-methods/theory/capacity'
import {
  choice,
  ControlRow,
  Curve,
  Figure,
  Handle,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'

const START: [number, number][] = [
  [-0.6, -0.4],
  [0.7, -0.5],
  [0.1, 0.7],
  [0.6, 0.5],
  [-0.7, 0.4],
  [0, -0.9],
  [-0.2, 0.1],
  [0.9, 0.1],
]
const clamp = (v: number) => Math.max(-1.1, Math.min(1.1, v))

/**
 * The projection of a class on a point set the reader moves: every labelling of the first n points, whether the class
 * realises it, and the number realised on the first k points for k = 1, …, n beside 2^k.
 */
export function ProjectionCount() {
  const state = useFigureState({
    count: int(5, { ge: 1, le: 8, suggestions: [3, 4, 5, 8], label: 'points n' }),
    family: choice(
      [
        { value: 'half-planes', label: 'half-planes (VC dimension 3)' },
        { value: 'rectangles', label: 'axis-aligned rectangles (VC 4)' },
        { value: 'intervals', label: 'intervals on x₁ (VC 2)' },
      ],
      'half-planes',
      { label: 'class' },
    ),
  })
  const { count, family } = state
  const [moved, setMoved] = useState<[number, number][]>(START)
  const points = useMemo(() => moved.slice(0, count), [moved, count])
  const nested = useMemo(
    () => points.map((_, k) => shatteringTable(points.slice(0, k + 1), family as ShatterClass)),
    [points, family],
  )
  const table = nested[nested.length - 1]
  const [k, setK] = usePlayhead(table.labellings.length)
  const at = Math.min(k, table.labellings.length - 1)
  const y = table.labellings[at]
  const sizes = points.map((_, i) => i + 1)
  const realised = nested.map((t) => t.count)
  const all = nested.map((t) => t.labellings.length)

  const x1 = useAxis({ label: 'x₁', range: [-1.2, 1.2] })
  const x2 = useAxis({ label: 'x₂', range: [-1.2, 1.2], equal: x1 })
  const nAxis = useAxis({ label: 'first k points', range: [0.5, 8.5] })
  const cAxis = useAxis({ label: 'labellings', range: [1, 300], log: true })
  const of = (sign: number) => {
    const idx = points.flatMap((_, i) => (y[i] === sign ? [i] : []))
    return { x: idx.map((i) => points[i][0]), y: idx.map((i) => points[i][1]) }
  }
  const negative = of(-1)
  const positive = of(1)
  return (
    <Figure
      title="The projection of a class on a sample"
      state={state}
      defaultSize="M"
      controls={
        <ControlRow label="labellings">
          <Player
            className="col-span-full"
            value={at}
            onChange={setK}
            count={table.labellings.length}
            label="labelling"
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="realised on n points" value={`${table.count} of ${table.labellings.length}`} />
          <Readout label="shattered" value={table.shattered ? 'yes' : 'no'} />
          <Readout label="labelling shown" value={table.realised[at] ? 'realised' : 'not realised'} />
        </>
      }
      caption="Left: the points, coloured by the labelling the player is on (positive in slot 1, negative in slot 0); drag any point. Right: the number of labellings the class realises on the first k points, beside all 2ᵏ. The engine's shatteringTable decides each labelling exactly (a feasibility linear program for half-planes, the bounding box of the positives for rectangles and intervals). The realised count follows 2ᵏ up to the VC dimension and then falls away. Making three points collinear removes labellings: the growth function is the count for the best-placed points."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} title={`labelling ${at}: ${table.realised[at] ? 'realised' : 'not realised'}`}>
          {negative.x.length > 0 && <Points name="negative" {...negative} slot={0} size={22} />}
          {positive.x.length > 0 && <Points name="positive" {...positive} slot={1} size={22} />}
          {points.map((q, i) => (
            <Handle
              key={i}
              kind="point"
              at={q}
              label={`${i + 1}`}
              onDrag={(v) =>
                setMoved((m) => m.map((old, j): [number, number] => (j === i ? [clamp(v[0]), clamp(v[1])] : old)))
              }
            />
          ))}
        </Plot>
        <Plot x={nAxis} y={cAxis} title="labellings realised on the first k points">
          <Curve name="all labellings, 2ᵏ" x={sizes} y={all} dashed />
          <Curve name="realised by the class" x={sizes} y={realised} slot={2} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
