import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { moons, pinwheel } from 'aifn-methods/data/synthetic'
import { child, standardNormals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { codePrefixTree, residualQuantiser, rqDecode, rqEncode } from 'aifn-compute/numerics/neighbours'

const N = 400
const SPAN = 3
const PANEL = 300

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

function points(kind: string): number[][] {
  const s = stream(`rq-vae/stages/${kind}`)
  let flat: ArrayLike<number>
  if (kind === 'moons') flat = toFlat(moons(s, { n: N, noise: 0.08 }).x)
  else if (kind === 'pinwheel') flat = toFlat(pinwheel(s, { n: N, arms: 4, noise: 0.05 }).x)
  else flat = standardNormals(child(s, 'gaussian'), 2 * N)
  return Array.from({ length: N }, (_, i) => [flat[2 * i], flat[2 * i + 1]])
}

/**
 * Residual quantisation of fixed 2-d points, with no network: one k-means codebook per stage, each trained on the
 * residuals the stages before it leave. The reconstruction after d stages is the sum of the first d chosen codewords.
 */
export function ResidualStages() {
  const state = useFigureState({
    data: choice(
      [
        { value: 'gaussian', label: 'standard normal' },
        { value: 'pinwheel', label: 'pinwheel (4 arms)' },
        { value: 'moons', label: 'two moons' },
      ],
      'pinwheel',
      { label: 'data' },
    ),
    codewords: choice([2, 3, 4, 8], 4, { label: 'codewords per stage K' }),
    levels: int(3, { ge: 1, le: 5, label: 'stages D' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { data, codewords, levels, seed } = state

  const x = useMemo(() => points(data), [data])
  const fit = useComputed(() => {
    const rq = residualQuantiser(x, { levels, codewords, stream: stream(`rq-vae/stages/fit/${seed}`) })
    const codes = rqEncode(rq, x)
    const codeRows = toFlat(codes)
    const byStage = Array.from({ length: levels }, (_, d) => {
      const r = toFlat(rqDecode(rq, codes, d + 1))
      return Array.from({ length: N }, (_, i) => [r[2 * i], r[2 * i + 1]])
    })
    const tree = codePrefixTree(codes)
    const prefixes = Array.from({ length: levels }, (_, d) => tree.filter((n) => n.depth === d + 1).length)
    return {
      first: Array.from({ length: N }, (_, i) => codeRows[i * levels]),
      byStage,
      prefixes,
      distortion: Array.from(rq.distortionByLevel),
    }
  }, [x, levels, codewords, seed])
  const f = fit.value

  const key = `${data}|${codewords}|${levels}|${seed}`
  const [pos, setPos] = useState({ key, stage: 0 })
  const stage = pos.key === key ? Math.min(pos.stage, levels - 1) : 0

  const raw = useMemo(() => ({ x: x.map((p) => p[0]), y: x.map((p) => p[1]) }), [x])
  // The reconstructions after the stage played, one series per stage-1 codeword, so a cluster keeps its colour as it
  // splits at later stages.
  const groups = useMemo(
    () =>
      Array.from({ length: codewords }, (_, k) => {
        const rows = f.byStage[stage].filter((_, i) => f.first[i] === k)
        return { x: rows.map((p) => p[0]), y: rows.map((p) => p[1]) }
      }),
    [f, stage, codewords],
  )
  const levelsAxis = useMemo(() => Array.from({ length: levels }, (_, d) => d + 1), [levels])

  const px = useAxis({ label: 'x₁', range: [-SPAN, SPAN] })
  const py = useAxis({ label: 'x₂', range: [-SPAN, SPAN], equal: px })
  const dx = useAxis({ label: 'stages used d', range: [0.5, levels + 0.5] })
  const dy = useAxis({ label: 'mean squared error after d stages', log: true })

  return (
    <Figure
      title="Residual quantisation, stage by stage"
      state={state}
      defaultSize="L"
      caption={`Residual quantisation of ${N} fixed points (grey), with no network. Stage 1 runs k-means with K codewords on the points; stage d runs k-means on the residuals the first d − 1 stages leave. The reconstruction after d stages is the sum of the first d chosen codewords (coloured dots, one colour per stage-1 codeword). Play the stages: after stage 1 the points snap to at most K places, after stage 2 to at most K², and each cluster splits inside its stage-1 colour, because later stages only add corrections. Right: the mean squared error after each stage, on a log scale, which never rises. The readouts count the distinct code prefixes after the stage played, out of at most Kᵈ.`}
      controls={
        <Player
          value={stage}
          onChange={(s) => setPos({ key, stage: s })}
          count={levels}
          label="stage"
          format={(s) => `stage ${s + 1} of ${levels}`}
        />
      }
      readouts={
        <>
          <Readout label="distinct prefixes" value={`${f.prefixes[stage]} of ≤ ${codewords ** (stage + 1)}`} />
          <Readout label={`mean squared error after stage ${stage + 1}`} value={formatNumber(f.distortion[stage])} />
          <Readout label="bits per point" value={formatNumber((stage + 1) * Math.log2(codewords))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={px} y={py}>
            <Points name="data" x={raw.x} y={raw.y} muted dense />
            {groups.map((g, k) => (
              <Points key={k} name={`stage-1 codeword ${k + 1}`} x={g.x} y={g.y} slot={k} size={6} />
            ))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={dx} y={dy}>
            <Curve name="error after d stages" x={levelsAxis} y={f.distortion} emphasis />
            <Points name="stage played" x={[stage + 1]} y={[f.distortion[stage]]} emphasis size={9} live />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
