import { describe, expect, it } from 'vitest'
import { LANE_OFFSET, pathMoves } from './grid'

type Cell = readonly [number, number]

/** The sideways offset of a horizontal arrow from its row. */
const offset = (v: { from: [number, number] }, row = 0) => +(v.from[1] - row).toFixed(6)
const rightward = (v: { from: [number, number]; to: [number, number] }) => v.to[0] > v.from[0]

describe('pathMoves', () => {
  it('aggregates moves by directed edge and labels counts above 1', () => {
    const cells: Cell[] = [
      [0, 0],
      [1, 0],
      [2, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ]
    const v = pathMoves(cells, 5)
    // 0→1, 1→2 (twice), 2→1, 2→3: four arrows.
    expect(v).toHaveLength(4)
    const twice = v.find((m) => m.label === '2')!
    expect(twice.labelAt).toBe('middle')
    expect(rightward(twice)).toBe(true)
    expect(v.filter((m) => m.label !== undefined)).toHaveLength(1)
    expect(twice.width!).toBeGreaterThan(v.find((m) => m.label === undefined)!.width!)
  })

  it('centres an edge walked one way and splits an edge walked both ways into two lanes', () => {
    const cells: Cell[] = [
      [0, 0],
      [1, 0],
      [2, 0],
      [1, 0],
    ]
    const v = pathMoves(cells, 3)
    expect(v).toHaveLength(3)
    const oneWay = v.find((m) => m.from[0] < 1)!
    expect(offset(oneWay)).toBe(0)
    const both = v.filter((m) => m.from[0] > 1 || m.to[0] > 1)
    expect(both).toHaveLength(2)
    expect(both.map((m) => Math.abs(offset(m)))).toEqual([LANE_OFFSET, LANE_OFFSET])
    expect(offset(both[0])).toBe(-offset(both[1]))
    expect(both.map(rightward).sort()).toEqual([false, true])
  })

  it('skips a move that stays in its cell (a wall bump)', () => {
    const cells: Cell[] = [
      [0, 0],
      [0, 0],
      [0, 1],
      [0, 1],
    ]
    const v = pathMoves(cells, 3)
    expect(v).toHaveLength(1)
    expect(v[0].from[0]).toBeCloseTo(0)
    expect(v[0].to[1]).toBeGreaterThan(v[0].from[1])
  })

  it('draws the arrow holding the latest move in ink and last, and stops at the step', () => {
    const cells: Cell[] = [
      [0, 0],
      [1, 0],
      [0, 0],
      [1, 0],
      [2, 0],
    ]
    const v = pathMoves(cells, 3, { slot: 4 })
    // Up to step 3: 0→1 twice (latest), 1→0 once.
    expect(v).toHaveLength(2)
    expect(v[1].slot).toBeUndefined()
    expect(v[1].label).toBe('2')
    expect(rightward(v[1])).toBe(true)
    expect(v[0].slot).toBe(4)
    expect(pathMoves(cells, 99, { inkLatest: false }).every((m) => m.slot === 1)).toBe(true)
    expect(pathMoves(cells, 0)).toEqual([])
  })
})
