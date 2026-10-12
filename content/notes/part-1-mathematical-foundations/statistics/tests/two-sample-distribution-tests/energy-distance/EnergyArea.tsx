import { useMemo } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import {
  asRows,
  distanceMatrix,
  energy,
  matrixPermutationTest,
  oneDimensionalPair,
  SHIFT_LABELS,
  type ShiftKind,
} from '../_shared/twoSample'

const LO = -4.5
const HI = 4.5

/**
 * The two empirical cdfs on the pooled points, as step functions: at each pooled point t (sorted), F̂(t) and Ĝ(t),
 * the fractions of each sample at or below t.
 */
function stepCdfs(x: readonly number[], y: readonly number[]) {
  const pooled = [...x.map((v) => ({ v, a: 1 })), ...y.map((v) => ({ v, a: 0 }))].sort((p, q) => p.v - q.v)
  let fx = 0
  let fy = 0
  const t: number[] = []
  const F: number[] = []
  const G: number[] = []
  for (const p of pooled) {
    if (p.a) fx += 1 / x.length
    else fy += 1 / y.length
    t.push(p.v)
    F.push(fx)
    G.push(fy)
  }
  return { t, F, G }
}

/** A step function that is `values[i]` on [t[i], t[i + 1]), drawn from LO to HI, starting at `start`. */
function stepCurve(t: readonly number[], values: readonly number[], start: number) {
  const x: number[] = [LO, t[0]]
  const y: number[] = [start, start]
  t.forEach((ti, i) => {
    const next = i + 1 < t.length ? t[i + 1] : HI
    x.push(ti, next)
    y.push(values[i], values[i])
  })
  return { x, y }
}

/**
 * Energy distance in one dimension: twice the area under (F̂ − Ĝ)², the squared gap between the two empirical cdfs,
 * which equals the distance form 2 mean|x − y| − mean|x − x′| − mean|y − y′| exactly.
 */
export function EnergyArea() {
  const state = useFigureState({
    kind: choice<ShiftKind>(
      (Object.keys(SHIFT_LABELS) as ShiftKind[]).map((value) => ({ value, label: SHIFT_LABELS[value] })),
      'scale',
      { label: 'how the second sample differs' },
    ),
    delta: float(0.6, { min: 0, max: 1, step: 0.01, label: 'strength δ' }),
    n: int(60, { min: 5, max: 300, label: 'points in each sample', suggestions: [20, 60, 300] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => oneDimensionalPair(state.seed, state.n, state.n, state.kind, state.delta),
    [state.seed, state.n, state.kind, state.delta],
  )
  const cdf = useMemo(() => stepCdfs(data.x, data.y), [data])
  const area = useMemo(() => {
    let a = 0
    for (let i = 0; i + 1 < cdf.t.length; i++) a += (cdf.F[i] - cdf.G[i]) ** 2 * (cdf.t[i + 1] - cdf.t[i])
    return a
  }, [cdf])
  const gap = useMemo(() => Math.max(...cdf.F.map((f, i) => Math.abs(f - cdf.G[i]))), [cdf])

  const test = useComputed(
    () => {
      const D = distanceMatrix(asRows([...data.x, ...data.y]))
      return matrixPermutationTest(stream(state.seed + 1), D, data.x.length, energy, 499)
    },
    [data, state.seed],
    { mode: 'release' },
  )

  const curves = useMemo(
    () => ({
      F: stepCurve(cdf.t, cdf.F, 0),
      G: stepCurve(cdf.t, cdf.G, 0),
      sq: stepCurve(
        cdf.t,
        cdf.F.map((f, i) => (f - cdf.G[i]) ** 2),
        0,
      ),
    }),
    [cdf],
  )

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'empirical cdf', range: [0, 1] })
  const xb = useAxis({ label: 'x', range: [LO, HI] })
  const yb = useAxis({ label: '(F̂ − Ĝ)²', range: [0, undefined] })
  return (
    <Figure
      title="Energy distance as an area"
      state={state}
      caption="Left: the empirical cdfs of a sample from N(0, 1) and a sample from an alternative. Right: the squared gap between them. Twice the shaded area is the energy distance, and the readouts check that it equals the distance form, an average of absolute differences between points. Kolmogorov–Smirnov uses only the largest gap; the energy distance adds up every gap, so a change of spread, which opens two moderate gaps on either side, counts twice."
      readouts={
        <>
          <Readout label="2 mean|x − y| − mean|x − x′| − mean|y − y′|" value={formatNumber(test.value.observed)} />
          <Readout label="2 × area under (F̂ − Ĝ)²" value={formatNumber(2 * area)} />
          <Readout label="largest gap (KS D)" value={formatNumber(gap)} />
          <Readout label="permutation p-value" value={formatNumber(test.value.pValue)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={280}>
          <Curve name="F̂, first sample" x={curves.F.x} y={curves.F.y} slot={0} />
          <Curve name="Ĝ, second sample" x={curves.G.x} y={curves.G.y} slot={1} />
        </Plot>
        <Plot x={xb} y={yb} height={280}>
          <Area name="(F̂ − Ĝ)²" x={curves.sq.x} y={curves.sq.y} slot={2} opacity={0.35} />
        </Plot>
      </div>
    </Figure>
  )
}
