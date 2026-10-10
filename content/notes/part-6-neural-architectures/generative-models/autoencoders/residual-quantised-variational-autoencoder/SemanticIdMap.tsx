import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { moons, pinwheel } from 'aifn-methods/data/synthetic'
import { child, standardNormals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { residualQuantiser, rqDecode, rqEncode } from 'aifn-compute/numerics/neighbours'

const N = 400
const SPAN = 3
const CELLS = 121
const GRID = Array.from({ length: CELLS }, (_, i) => -SPAN + (2 * SPAN * i) / (CELLS - 1))
const GRID_ROWS = GRID.flatMap((y) => GRID.map((x) => [x, y]))
const MAX_STAGES = 5

function points(kind: string): number[][] {
  const s = stream(`rq-vae/ids/${kind}`)
  let flat: ArrayLike<number>
  if (kind === 'moons') flat = toFlat(moons(s, { n: N, noise: 0.08 }).x)
  else if (kind === 'pinwheel') flat = toFlat(pinwheel(s, { n: N, arms: 4, noise: 0.05 }).x)
  else flat = standardNormals(child(s, 'gaussian'), 2 * N)
  return Array.from({ length: N }, (_, i) => [flat[2 * i], flat[2 * i + 1]])
}

/** Focus options for up to four codewords; a codeword the current K lacks counts as no focus. */
const FOCUS = [
  { value: -1, label: 'all' },
  ...Array.from({ length: 4 }, (_, j) => ({ value: j, label: j < 2 ? `codeword ${j}` : `codeword ${j} (K > ${j})` })),
]

/** Row i's code, D digits from a flat [n × D] array. */
const codeAt = (flat: ArrayLike<number>, i: number, D: number) => Array.from({ length: D }, (_, d) => flat[i * D + d])

/** Whether a code lies in the focus: its first digit is f1 (or f1 < 0) and its second is f2 (or f2 < 0). */
const inFocus = (c: number[], f1: number, f2: number) =>
  (f1 < 0 || c[0] === f1) && (f2 < 0 || c.length < 2 || c[1] === f2)

/**
 * The partition of the plane by a residual quantiser's codes. Every point of the plane is encoded as the data would be,
 * and coloured by the digits of its code at the stages chosen, read as one base-K number, so that codes sharing a
 * prefix get nearby shades.
 */
export function SemanticIdMap() {
  const state = useFigureState({
    data: choice(
      [
        { value: 'pinwheel', label: 'pinwheel (4 arms)' },
        { value: 'moons', label: 'two moons' },
        { value: 'gaussian', label: 'standard normal' },
      ],
      'pinwheel',
      { label: 'data' },
    ),
    codewords: choice([2, 3, 4], 2, { label: 'codewords per stage K' }),
    levels: int(4, { ge: 1, le: MAX_STAGES, label: 'stages D' }),
    use1: setting(true, 'colour by stage 1'),
    use2: setting(true, 'colour by stage 2'),
    use3: setting(false, 'colour by stage 3'),
    use4: setting(false, 'colour by stage 4'),
    use5: setting(false, 'colour by stage 5'),
    edges: int(4, { ge: 0, le: MAX_STAGES, label: 'cell edges down to stage' }),
    focus1: choice(FOCUS, -1, { label: 'focus: stage-1 codeword' }),
    focus2: choice(FOCUS, -1, { label: 'focus: stage-2 codeword' }),
    px: slider(-SPAN, SPAN, 1.2, { step: 0.01, onChart: true }),
    py: slider(-SPAN, SPAN, 0.9, { step: 0.01, onChart: true }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { data, codewords, levels, edges, px, py, seed } = state

  const x = useMemo(() => points(data), [data])
  // The quantiser and the code of every grid cell, refitted only when the data, K, D or the seed change. Everything
  // below reads K and D from the fit, so a frame drawn while a refit is pending stays consistent with its codes.
  const fit = useComputed(() => {
    const rq = residualQuantiser(x, { levels, codewords, stream: stream(`rq-vae/ids/fit/${seed}`) })
    const grid = Int32Array.from(toFlat(rqEncode(rq, GRID_ROWS)))
    return { rq, grid, data: toFlat(rqEncode(rq, x)), K: codewords, D: levels }
  }, [x, codewords, levels, seed])
  const { rq, grid, K, D } = fit.value
  const f1 = state.focus1 < K ? state.focus1 : -1
  const f2 = state.focus2 < K ? state.focus2 : -1
  const use = [state.use1, state.use2, state.use3, state.use4, state.use5].slice(0, D)

  const chosenKey = use.flatMap((on, d) => (on ? [d] : [])).join(',')
  const chosen = useMemo(() => (chosenKey ? chosenKey.split(',').map(Number) : []), [chosenKey])
  const classes = K ** chosen.length
  // The colour of each cell: the chosen digits of its code as one base-K number, most significant first. Cells outside
  // the focus are left empty.
  const colour = useMemo(() => {
    const value = (c: number[]) => chosen.reduce((v, d) => v * K + c[d], 0)
    return GRID.map((_, r) =>
      GRID.map((_, col) => {
        const c = codeAt(grid, r * CELLS + col, D)
        return inFocus(c, f1, f2) && chosen.length > 0 ? value(c) : NaN
      }),
    )
  }, [grid, chosen, K, D, f1, f2])
  // The cell edges of the code prefixes down to `edges` stages, as short segments between neighbouring grid cells whose
  // prefixes differ. A second raster cannot draw them: a raster paints opaquely and would hide the shading.
  const cellEdges = useMemo(() => {
    const depth = Math.min(edges, D)
    if (depth === 0) return []
    const label = (r: number, col: number) => {
      const c = codeAt(grid, r * CELLS + col, D)
      return inFocus(c, f1, f2) ? c.slice(0, depth).reduce((v, k) => v * K + k, 0) : -1
    }
    const h = GRID[1] - GRID[0]
    const out: { from: [number, number]; to: [number, number] }[] = []
    for (let r = 0; r < CELLS; r++)
      for (let col = 0; col < CELLS; col++) {
        const here = label(r, col)
        if (col + 1 < CELLS && here !== label(r, col + 1)) {
          const xm = GRID[col] + h / 2
          out.push({ from: [xm, GRID[r] - h / 2], to: [xm, GRID[r] + h / 2] })
        }
        if (r + 1 < CELLS && here !== label(r + 1, col)) {
          const ym = GRID[r] + h / 2
          out.push({ from: [GRID[col] - h / 2, ym], to: [GRID[col] + h / 2, ym] })
        }
      }
    return out
  }, [grid, edges, K, D, f1, f2])

  const shown = useMemo(() => {
    const on = { x: [] as number[], y: [] as number[] }
    const off = { x: [] as number[], y: [] as number[] }
    x.forEach((p, i) => {
      const into = inFocus(codeAt(fit.value.data, i, D), f1, f2) ? on : off
      into.x.push(p[0])
      into.y.push(p[1])
    })
    return { on, off }
  }, [x, fit.value, f1, f2, D])

  // The probe's code and its partial sums: the reconstruction after each stage, drawn as one arrow per codeword.
  const probe = useMemo(() => {
    const c = toFlat(rqEncode(rq, [[px, py]]))
    const sums = Array.from({ length: D }, (_, d) => toFlat(rqDecode(rq, [Array.from(c)], d + 1)))
    const arrows: Vector[] = sums.map((s, d) => ({
      from: d === 0 ? [0, 0] : [sums[d - 1][0], sums[d - 1][1]],
      to: [s[0], s[1]],
      label: `stage ${d + 1}`,
      labelAt: 'middle',
    }))
    const errors = sums.map((s) => (s[0] - px) ** 2 + (s[1] - py) ** 2)
    const shares = Array.from({ length: D }, (_, d) => {
      let n = 0
      for (let i = 0; i < N; i++) {
        let same = true
        for (let j = 0; j <= d && same; j++) same = fit.value.data[i * D + j] === c[j]
        if (same) n++
      }
      return n
    })
    return { code: Array.from(c), arrows, errors, shares }
  }, [rq, px, py, D, fit.value])

  const ax = useAxis({ label: 'x₁', range: [-SPAN, SPAN] })
  const ay = useAxis({ label: 'x₂', range: [-SPAN, SPAN], equal: ax })
  const id = `(${probe.code.join(', ')})`

  return (
    <Figure
      title="Semantic IDs as a map"
      state={state}
      defaultSize="L"
      caption={`Residual quantisation of ${N} points (grey), with D stages of K codewords each, fitted by k-means stage by stage. Every point of the plane is encoded as a data point would be, so its code (its semantic ID) is a tuple of D digits from 0 to K − 1; with K = 2 the IDs are bit strings. The shading reads the digits at the stages switched on as one base-K number, most significant first: with one stage on, each codeword gets its own colour; with several, IDs that share their leading digits get nearby shades. The ink lines are the cell edges of the code prefixes down to the stage chosen. Colour by stage 1 alone to see the coarse cells. Colour by stage 2 alone to see the same correction repeat inside every stage-1 cell, because stage 2 quantises the residual whatever stage 1 chose. Focus on one stage-1 codeword, and then one stage-2 codeword, to see how the next stage splits that branch; points outside the focus are drawn faint and the cells outside it are left blank. Drag the probe: the arrows from the origin are its chosen codewords, one per stage, and they end at its reconstruction.`}
      readouts={
        <>
          <Readout label="probe's semantic ID" value={id} />
          <Readout label="probe's error after each stage" value={probe.errors.map((e) => formatNumber(e)).join(', ')} />
          <Readout label="data points sharing its prefix of length 1, 2, …" value={probe.shares.join(', ')} />
          <Readout label="colours in use" value={chosen.length === 0 ? 'none' : `${classes}`} />
        </>
      }
    >
      <Plot x={ax} y={ay}>
        {chosen.length > 0 && (
          <Raster
            name="semantic ID"
            x={GRID}
            y={GRID}
            z={colour}
            scale={chosen.length === 1 ? 'categorical' : 'sequential'}
            // The scale starts one step below the first ID, so that ID 0 is not drawn white.
            range={chosen.length === 1 ? undefined : [-1, classes - 1]}
            colorBar={false}
            fillOpacity={0.55}
            valueLabel="chosen digits as a number"
          />
        )}
        {cellEdges.length > 0 && <Segments name="cell edges" segments={cellEdges} emphasis width={1.2} />}
        {/* An empty dense Points layer makes ECharts throw, and nothing is outside the focus until one is chosen. */}
        {shown.off.x.length > 0 && (
          <Points name="data outside the focus" x={shown.off.x} y={shown.off.y} muted dense thin />
        )}
        <Points name="data" x={shown.on.x} y={shown.on.y} muted size={4} />
        <Vectors vectors={probe.arrows} />
        <Handle {...state.handle(['px', 'py'], { label: `probe ${id}` })} />
      </Plot>
    </Figure>
  )
}
