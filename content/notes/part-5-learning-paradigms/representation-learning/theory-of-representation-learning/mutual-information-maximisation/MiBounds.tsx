import { useMemo } from 'react'
import {
  Annotation,
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  correlationForMutualInformation,
  donskerVaradhanBound,
  infoNceBound,
  nwjBound,
} from 'aifn-compute/probability/information'
import { gaussianPairLogRatios } from 'aifn-methods/data/synthetic'

/** True mutual information values along the x-axis, in nats. */
const GRID = Array.from({ length: 17 }, (_, i) => i * 0.5)
/** Batches drawn at each value, for the mean and the spread. */
const REPEATS = 6
const ESTIMATORS = ['InfoNCE', 'NWJ', 'DV (MINE)'] as const

/** The three estimates from one batch of n Gaussian pairs scored by the optimal critic. */
function estimates(seed: number, at: number, r: number, mi: number, n: number): number[] {
  const batch = gaussianPairLogRatios(child(stream(`mi-bounds/${seed}`), at, r), {
    n,
    rho: correlationForMutualInformation(mi),
  })
  // NWJ's optimal critic is the log density ratio plus one.
  const shifted = batch.scores.map((v) => v + 1)
  return [infoNceBound(batch.scores, n), nwjBound(shifted, n), donskerVaradhanBound(batch.scores, n)]
}

/** Mean and standard deviation of each estimator over the repeated batches. */
function summarise(rows: number[][]): { mean: number[]; sd: number[] } {
  const mean = ESTIMATORS.map((_, k) => rows.reduce((a, r) => a + r[k], 0) / rows.length)
  const sd = ESTIMATORS.map((_, k) =>
    Math.sqrt(rows.reduce((a, r) => a + (r[k] - mean[k]) ** 2, 0) / Math.max(1, rows.length - 1)),
  )
  return { mean, sd }
}

/**
 * InfoNCE, NWJ and Donsker-Varadhan estimates of the mutual information of a correlated Gaussian pair, each with the
 * optimal critic, against the true value. InfoNCE flattens at log N; NWJ is right on average but its spread grows
 * exponentially with the true value.
 */
export function MiBounds() {
  const state = useFigureState({
    n: int(64, { ge: 4, le: 512, suggestions: [16, 64, 256, 512], label: 'batch size N' }),
    mi: float(3, { min: 0, max: 8, step: 0.1, label: 'true mutual information (nats)', onChart: true }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, mi, seed } = state

  const run = useComputed(
    () =>
      GRID.map((value, at) => summarise(Array.from({ length: REPEATS }, (_, r) => estimates(seed, at, r, value, n)))),
    [n, seed],
    { mode: 'release' },
  )
  const curves = run.value

  // The estimates at the dragged value, from more batches than the curves use.
  const here = useMemo(
    () => summarise(Array.from({ length: 4 * REPEATS }, (_, r) => estimates(seed, 1000, r, mi, n))),
    [seed, mi, n],
  )

  const x = useAxis({ label: 'true mutual information I (nats)', range: [0, 8] })
  const y = useAxis({ label: 'estimate (nats)', range: [-2, 10] })
  const logN = Math.log(n)

  return (
    <Figure
      title="Lower bounds on mutual information against the truth"
      state={state}
      caption={`Pairs (x, y) are standard Gaussians with correlation ρ, chosen so the true mutual information I = −½ log(1 − ρ²) runs along the x-axis. Every estimator uses the optimal critic, the exact log density ratio, so all of its error comes from having N samples. Lines are means over ${REPEATS} batches of N pairs; bands are ± one standard deviation. InfoNCE stays below log N (horizontal line) at any I. NWJ is right on average, but its spread grows like e^I. Drag the vertical line to read the estimates at one value of I; change N to move the InfoNCE ceiling.`}
      readouts={
        <>
          <Readout label="log N" value={formatNumber(logN)} />
          {ESTIMATORS.map((name, k) => (
            <Readout
              key={name}
              label={`${name} at I = ${formatNumber(mi)}`}
              value={`${formatNumber(here.mean[k])} ± ${formatNumber(here.sd[k])}`}
            />
          ))}
        </>
      }
    >
      <Plot x={x} y={y} height={360}>
        <Curve name="true I" x={[0, 8]} y={[0, 8]} emphasis dashed />
        <Annotation y={logN} text="log N" muted dashed />
        {ESTIMATORS.map((name, k) => (
          <Area
            key={`${name}-band`}
            name={name}
            x={GRID}
            y={curves.map((c) => c.mean[k] + c.sd[k])}
            base={curves.map((c) => c.mean[k] - c.sd[k])}
            slot={k}
            opacity={0.15}
            line={false}
            stale={run.stale}
          />
        ))}
        {ESTIMATORS.map((name, k) => (
          <Curve
            key={name}
            name={name}
            x={GRID}
            y={curves.map((c) => c.mean[k])}
            slot={k}
            showPoints
            stale={run.stale}
          />
        ))}
        <Handle {...state.handle('mi', { label: 'I' })} />
      </Plot>
    </Figure>
  )
}
