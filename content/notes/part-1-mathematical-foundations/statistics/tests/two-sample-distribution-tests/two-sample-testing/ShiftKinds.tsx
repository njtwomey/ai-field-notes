import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Histogram,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalLogPdf } from 'aifn-compute/numerics/special'
import {
  allPValues,
  alternativeDensity,
  ONE_D_TESTS,
  oneDimensionalPair,
  SHIFT_LABELS,
  type ShiftKind,
} from '../_shared/twoSample'

const X = toFlat(linspace(-4.5, 4.5, 301))
const ALPHA = 0.05

/**
 * One pair of samples, from N(0, 1) and from an alternative that differs in its mean, its spread or only its shape,
 * with the p-value of each two-sample test on them.
 */
export function ShiftKinds() {
  const state = useFigureState({
    kind: choice<ShiftKind>(
      (Object.keys(SHIFT_LABELS) as ShiftKind[]).map((value) => ({ value, label: SHIFT_LABELS[value] })),
      'shape',
      { label: 'how the second sample differs' },
    ),
    delta: float(0.9, { min: 0, max: 1, step: 0.01, label: 'strength δ (0: no difference)' }),
    n: int(150, { min: 10, max: 300, label: 'points in each sample', suggestions: [30, 150, 300] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => oneDimensionalPair(state.seed, state.n, state.n, state.kind, state.delta),
    [state.seed, state.n, state.kind, state.delta],
  )
  // The permutation tests are the slow part: recompute when a slider is released.
  const tests = useComputed(() => allPValues(stream(state.seed + 1), data.x, data.y, 199), [data, state.seed], {
    mode: 'release',
  })
  const p = tests.value
  const curves = useMemo(
    () => ({
      p: X.map((x) => Math.exp(normalLogPdf(x))),
      q: X.map((x) => alternativeDensity(state.kind, state.delta, x)),
    }),
    [state.kind, state.delta],
  )

  const xa = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const ya = useAxis({ label: 'density', range: [0, undefined] })
  return (
    <Figure
      title="Three ways two samples can differ"
      state={state}
      caption="Histograms of a sample from N(0, 1) and a sample from an alternative, with both true densities. The readouts give each test's p-value; a p-value below 0.05 rejects the hypothesis that the two samples share a distribution. The t-test compares means, so it misses a change of spread or shape. Mann–Whitney compares which sample tends to be larger, so it misses them too. The Kolmogorov–Smirnov, energy, MMD and classifier tests compare whole distributions. Set δ to 0 to see the tests reject about 5% of the time when nothing differs."
      readouts={
        <>
          {ONE_D_TESTS.map((t) => (
            <Readout key={t} label={t} value={`p = ${formatNumber(p[t])}${p[t] <= ALPHA ? ' · reject' : ''}`} />
          ))}
        </>
      }
    >
      <Plot x={xa} y={ya} height={300}>
        <Histogram name="sample from N(0, 1)" values={data.x} bins={30} range={[-4.5, 4.5]} slot={0} opacity={0.3} />
        <Histogram
          name="sample from the alternative"
          values={data.y}
          bins={30}
          range={[-4.5, 4.5]}
          slot={1}
          opacity={0.3}
        />
        <Curve name="N(0, 1)" x={X} y={curves.p} slot={0} />
        <Curve name="alternative" x={X} y={curves.q} slot={1} />
      </Plot>
    </Figure>
  )
}
