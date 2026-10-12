import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  Rug,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { ksTest } from 'aifn-compute/probability/tests'
import { oneDimensionalPair, SHIFT_LABELS, type ShiftKind } from '../_shared/twoSample'

const LO = -4.5
const HI = 4.5

/** The empirical cdf as a step curve: the fraction of the sample at or below x, from LO to HI. */
function steps(sample: readonly number[]) {
  const s = [...sample].sort((a, b) => a - b)
  const x: number[] = [LO]
  const y: number[] = [0]
  s.forEach((v, i) => {
    x.push(v, v)
    y.push(i / s.length, (i + 1) / s.length)
  })
  x.push(HI)
  y.push(1)
  return { x, y }
}

const ecdfAt = (sample: readonly number[], t: number) => sample.filter((v) => v <= t).length / sample.length

/**
 * The two-sample Kolmogorov–Smirnov statistic: the largest vertical gap between the two empirical distribution
 * functions, with its p-value from the engine's exact null law.
 */
export function EcdfGap() {
  const state = useFigureState({
    kind: choice<ShiftKind>(
      (Object.keys(SHIFT_LABELS) as ShiftKind[]).map((value) => ({ value, label: SHIFT_LABELS[value] })),
      'mean',
      { label: 'how the second sample differs' },
    ),
    delta: float(0.4, { min: 0, max: 1, step: 0.01, label: 'strength δ' }),
    n: int(60, { min: 5, max: 500, label: 'points in the first sample', suggestions: [20, 60, 500] }),
    m: int(60, { min: 5, max: 500, label: 'points in the second sample', suggestions: [20, 60, 500] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => oneDimensionalPair(state.seed, state.n, state.m, state.kind, state.delta),
    [state.seed, state.n, state.m, state.kind, state.delta],
  )
  const test = useMemo(() => ksTest(data.x, data.y), [data])
  const curves = useMemo(() => ({ f: steps(data.x), g: steps(data.y) }), [data])
  const at = test.location
  const gap = [{ from: [at, ecdfAt(data.x, at)] as const, to: [at, ecdfAt(data.y, at)] as const }]

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'fraction of the sample ≤ x', range: [0, 1] })
  return (
    <Figure
      title="The largest gap between two empirical cdfs"
      state={state}
      caption="The empirical distribution functions of a sample from N(0, 1) and a sample from an alternative. The black segment is the Kolmogorov–Smirnov statistic D, the largest vertical distance between them. A mean shift opens a gap in the middle, where the cdfs are steep, and D finds it easily; a change of spread or shape opens smaller gaps of opposite sign on either side, which a single maximum sees less well."
      readouts={
        <>
          <Readout label="D" value={formatNumber(test.statistic)} />
          <Readout label="at x" value={formatNumber(at)} />
          <Readout label="p-value" value={formatNumber(test.pValue)} />
          <Readout label="method" value={test.method} />
        </>
      }
    >
      <Plot x={xa} y={ya} height={300}>
        <Curve name="F̂, first sample" x={curves.f.x} y={curves.f.y} slot={0} />
        <Curve name="Ĝ, second sample" x={curves.g.x} y={curves.g.y} slot={1} />
        <Segments name="D" segments={gap} emphasis width={3} />
        <Rug name="first sample" values={data.x} slot={0} />
        <Rug name="second sample" values={data.y} slot={1} />
      </Plot>
    </Figure>
  )
}
