import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { selectiveScan, semiseparableMask, ssdChunked, ssdQuadratic } from 'aifn-compute/nn/sequence'

const T = 24
const N = 4
const P = 2
const STEPS = Array.from({ length: T }, (_, t) => t + 1)
const CENTRES = Array.from({ length: T }, (_, i) => i + 0.5)
const CHUNKS = Array.from({ length: T }, (_, i) => i + 1)

type Pattern = 'constant' | 'gated'
const PATTERNS = [
  { value: 'gated', label: 'input-dependent (a few resets)' },
  { value: 'constant', label: 'constant decay' },
] as const

/**
 * The 1-semiseparable mask of Mamba-2 built from per-token decays a_t, the chunked SSD algorithm that splits it into
 * diagonal blocks (computed as attention-like matrix products) and off-diagonal blocks (passed through chunk states),
 * and three computations of the same outputs: the quadratic form, the chunked algorithm and the engine's selective
 * scan.
 */
export function SemiseparableMask() {
  const state = useFigureState({
    pattern: choice<Pattern>(PATTERNS, 'gated', { label: 'decays' }),
    base: float(0.9, { min: 0.3, max: 1, step: 0.01, label: 'typical decay a' }),
    chunk: int(6, { min: 1, max: T, label: 'chunk length Q', onChart: true }),
    seed: int(2, { min: 0, max: 99, label: 'seed', suggestions: [1, 2, 3] }),
  })
  const { pattern, base, chunk, seed } = state

  const problem = useMemo(() => {
    const root = stream(seed)
    const rd = child(root, 'decays')
    const a = STEPS.map(() => {
      if (pattern === 'constant') return base
      const u = uniform(rd)
      return u < 0.15 ? 0.05 + 0.1 * uniform(rd) : Math.min(1, base + 0.08 * (uniform(rd) - 0.5))
    })
    const rx = child(root, 'inputs')
    const x = STEPS.map(() => Array.from({ length: P }, () => normal(rx)))
    const B = STEPS.map(() => Array.from({ length: N }, () => normal(rx) / Math.sqrt(N)))
    const C = STEPS.map(() => Array.from({ length: N }, () => normal(rx) / Math.sqrt(N)))
    const mask = toRows(semiseparableMask(a) as Tensor)
    const quadratic = toRows(ssdQuadratic(x, a, B, C) as Tensor)
    // The engine's selective scan with Δ_t = −log a_t, A = −1 and B_t / Δ_t (its Euler rule multiplies B by Δ_t).
    const delta = a.map((v) => -Math.log(v) || 1e-12)
    const scan = toFlat(
      selectiveScan(
        tensor(x),
        tensor(delta.map((d) => Array(P).fill(d))),
        tensor(Array.from({ length: P }, () => Array(N).fill(-1))),
        tensor(B.map((row, t) => row.map((b) => b / delta[t]))),
        tensor(C),
      ).outputs as Tensor,
    )
    const costs = CHUNKS.map((q) => ssdChunked(x, a, B, C, q).cost)
    return { a, x, B, C, mask, quadratic, scan, costs }
  }, [pattern, base, seed])

  const chunked = useMemo(() => {
    const run = ssdChunked(problem.x, problem.a, problem.B, problem.C, chunk)
    return { y: toRows(run.y as Tensor), cost: run.cost }
  }, [problem, chunk])
  const z = useMemo(() => problem.mask.map((row, j) => row.map((v, i) => (i <= j ? v : NaN))), [problem.mask])
  const grid = useMemo(() => {
    const segments: { from: [number, number]; to: [number, number] }[] = []
    for (let k = chunk; k < T; k += chunk) {
      segments.push({ from: [k, k], to: [k, T] })
      segments.push({ from: [0, k], to: [k, k] })
    }
    return segments
  }, [chunk])

  const quadY = problem.quadratic.map((r) => r[0])
  const chunkY = chunked.y.map((r) => r[0])
  const scanY = STEPS.map((_, t) => problem.scan[t * P])
  const errChunk = Math.max(...problem.quadratic.flatMap((r, t) => r.map((v, p) => Math.abs(v - chunked.y[t][p]))))
  const errScan = Math.max(
    ...problem.quadratic.flatMap((r, t) => r.map((v, p) => Math.abs(v - problem.scan[t * P + p]))),
  )
  const cost = chunked.cost

  const iAxis = useAxis({ label: 'source position i', range: [0, T] })
  const jAxis = useAxis({ label: 'output position j', range: [0, T], inverse: true, equal: iAxis })
  const tAxis = useAxis({ label: 'position t', range: [0.5, T + 0.5], integer: true })
  const aAxis = useAxis({ label: 'decay a_t', range: [0, 1] })
  const qAxis = useAxis({ label: 'chunk length Q', range: [1, T], integer: true })
  const cAxis = useAxis({ label: 'count', log: true })
  const yAxis = useAxis({ label: 'y_t (channel 1)', hold: 'union', key: `${seed}-${pattern}` })

  return (
    <Figure
      title="The semiseparable mask and its chunks"
      state={state}
      caption="Each entry L_ji of the mask is the product of the decays a_{i+1} ⋯ a_j between source i and output j. A reset (a small a_t) cuts every entry that spans it, so the mask falls to near zero below and to the left of that position. Drag the line labelled Q to change the chunk length. Diagonal blocks are computed as small attention-like matrix products; each off-diagonal block has rank N and is passed through the chunk states, one sequential step per chunk. Q = 1 is the pure recurrence (24 sequential steps), Q = 24 is the pure quadratic form (one step, the most multiplies). The quadratic form (dots), the chunked algorithm (line) and the engine's selective scan (dashed) give the same outputs."
      readouts={
        <>
          <Readout label="multiplies in diagonal blocks" value={formatNumber(cost.intraChunk)} />
          <Readout label="multiplies through chunk states" value={formatNumber(cost.stateProducts)} />
          <Readout label="sequential steps" value={`${cost.sequentialSteps}`} />
          <Readout label="max |chunked − quadratic|" value={errChunk.toExponential(1)} />
          <Readout label="max |selective scan − quadratic|" value={errScan.toExponential(1)} />
        </>
      }
    >
      <Plots rows={2} cols={2}>
        <Plot x={iAxis} y={jAxis}>
          <Raster x={CENTRES} y={CENTRES} z={z} scale="sequential" range={[0, 1]} valueLabel="L_ji" />
          <Segments segments={grid} emphasis width={1.5} />
          <Handle {...state.handle('chunk', { label: 'Q' })} />
        </Plot>
        <Plot x={tAxis} y={aAxis}>
          <Bars name="decay a_t" x={STEPS} y={problem.a} slot={0} />
        </Plot>
        <Plot x={qAxis} y={cAxis}>
          <Curve name="multiplies, diagonal blocks" x={CHUNKS} y={problem.costs.map((c) => c.intraChunk)} slot={0} />
          <Curve name="multiplies, chunk states" x={CHUNKS} y={problem.costs.map((c) => c.stateProducts)} slot={1} />
          <Curve name="sequential steps" x={CHUNKS} y={problem.costs.map((c) => c.sequentialSteps)} slot={2} dashed />
          <Annotation x={chunk} text={`Q = ${chunk}`} dashed />
        </Plot>
        <Plot x={tAxis} y={yAxis}>
          <Points name="quadratic form" x={STEPS} y={quadY} slot={0} />
          <Curve name="chunked" x={STEPS} y={chunkY} slot={1} />
          <Curve name="selective scan" x={STEPS} y={scanY} slot={2} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
