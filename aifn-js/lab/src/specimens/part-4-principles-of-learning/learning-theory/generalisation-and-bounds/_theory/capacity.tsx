/**
 * Shattering and Rademacher complexity on a point set the reader moves (`aifn-applied/theory/capacity`): every
 * labelling of the points, whether half-planes, rectangles or intervals realise it, and the empirical Rademacher
 * complexity of what they realise.
 */
import { useMemo, useState } from 'react'
import { empiricalRademacher, shatteringTable, type ShatterClass } from 'aifn-applied/theory/capacity'
import { stream } from 'aifn/foundation/random'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, int, row, useFigureState } from '@lab/state'
import { Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
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

export function ShatteringSpecimen() {
  const state = useFigureState({
    setup: row('1 · points and class', {
      count: int(3, { ge: 1, le: 8, suggestions: [2, 3, 4, 5], label: 'points' }),
      family: choice(
        [
          { value: 'half-planes', label: 'half-planes (VC dimension 3)' },
          { value: 'rectangles', label: 'axis-aligned rectangles (VC 4)' },
          { value: 'intervals', label: 'intervals on x₁ (VC 2)' },
        ],
        'half-planes',
        { label: 'class' },
      ),
    }),
  })
  const { count, family } = state.setup
  const [moved, setMoved] = useState<[number, number][]>(START)
  const points = moved.slice(0, count)
  const table = useMemo(() => shatteringTable(points, family as ShatterClass), [points, family])
  const rademacher = useMemo(
    () =>
      empiricalRademacher(
        stream('rademacher'),
        table.labellings.filter((_, i) => table.realised[i]),
        1000,
      ),
    [table],
  )
  const [k, setK] = usePlayhead(table.labellings.length)
  const at = Math.min(k, table.labellings.length - 1)
  const y = table.labellings[at]
  const x1 = useAxis({ label: 'x₁', range: [-1.2, 1.2] })
  const x2 = useAxis({ label: 'x₂', range: [-1.2, 1.2], equal: x1 })
  const dAxis = useAxis({ label: 'sign draws', range: [1, 1000], log: true })
  const rAxis = useAxis({ label: 'Rademacher estimate', range: [0, 1.2] })
  const of = (sign: number) => {
    const idx = points.flatMap((_, i) => (y[i] === sign ? [i] : []))
    return { x: idx.map((i) => points[i][0]), y: idx.map((i) => points[i][1]) }
  }
  const draws = useMemo(() => Float64Array.from({ length: 1000 }, (_, i) => i + 1), [])
  return (
    <Figure
      title="Shattering and Rademacher complexity"
      purpose="A class shatters a set of points when it realises all 2ⁿ of their labellings; the VC dimension is the largest set it can shatter, and the empirical Rademacher complexity, how well the realised labellings can match random signs, falls from 1 as soon as some labellings are out of reach."
      state={state}
      defaultSize="M"
      controls={
        <ControlRow label="2 · labellings">
          <Player
            className="col-span-full"
            value={at}
            onChange={setK}
            count={table.labellings.length}
            label="labelling"
          />
        </ControlRow>
      }
      readouts={{
        class: (
          <>
            <Readout label="realised" value={`${table.count} / ${table.labellings.length}`} />
            <Readout label="shattered" value={table.shattered ? 'yes' : 'no'} />
            <Readout label="labelling shown" value={table.realised[at] ? 'realisable' : 'not realisable'} />
          </>
        ),
        Rademacher: (
          <>
            <Readout label="estimate" value={`${fmt(rademacher.estimate)} ± ${fmt(rademacher.standardError, 1)}`} />
            <Readout label="Massart √(2 ln|H|/n)" value={fmt(rademacher.massart)} />
          </>
        ),
      }}
      caption="aifn shatteringTable enumerates the labellings (half-planes by a feasibility linear program, rectangles and intervals by the bounding box of the positives); empiricalRademacher averages max over the realised labellings of (1/n) Σ σᵢ hᵢ over 1000 random sign vectors. Drag the points; play through the labellings (positive in slot 1, negative in slot 0). Three points in general position are shattered by half-planes; four never are (the two crossing pairs cannot be split); collinear points lose labellings."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} title={`labelling ${at}: ${table.realised[at] ? 'realisable' : 'not realisable'}`}>
          <Points name="negative" {...of(-1)} slot={0} size={22} />
          <Points name="positive" {...of(1)} slot={1} size={22} />
          {points.map((q, i) => (
            <Handle
              key={i}
              kind="point"
              at={q}
              label={`${i + 1}`}
              onDrag={(v) =>
                setMoved((m) =>
                  m.map((old, j): [number, number] =>
                    j === i ? [Math.max(-1.1, Math.min(1.1, v[0])), Math.max(-1.1, Math.min(1.1, v[1]))] : old,
                  ),
                )
              }
            />
          ))}
        </Plot>
        <Plot x={dAxis} y={rAxis} title="Rademacher estimate as signs are drawn">
          <Curve name="estimate" slot={2} x={draws} y={rademacher.running} />
          <Curve name="Massart bound" x={[1, 1000]} y={[rademacher.massart, rademacher.massart]} emphasis dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
