import { call, Figure, Plot, Plots, Raster, useAxis, useComputed, type Task } from 'aifn-render'
import { toArray, type Tensor } from 'aifn-compute/foundation/tensor'

/** The design grid: group 1 fixed, group 2's size and its standard deviation relative to group 1's. */
const N1 = 20
const ALPHA = 0.05
const REPS = 1000
const LOG_RATIOS = Array.from({ length: 17 }, (_, i) => -2 + i / 4)
const N2S = Array.from({ length: 12 }, (_, j) => 5 * (j + 1))
const SEED = 'welch-t-test/type-i-error'

/**
 * Engine tensor operations as worker tasks, so the simulation reads as array code but runs in the compute worker.
 * Every argument is a number, a plain array or another task.
 */
const op =
  (address: string) =>
  (...args: unknown[]): Task<Tensor> =>
    call<Tensor>(address, ...args)
const [add, sub, mul, div, sqrt, abs, neg, square, less, mean, stack, reshape, tensor] = [
  'add',
  'sub',
  'mul',
  'div',
  'sqrt',
  'abs',
  'neg',
  'square',
  'less',
  'mean',
  'stack',
  'reshape',
  'tensor',
].map((name) => op(`foundation/tensor/${name}`))
const studentTCdf = op('numerics/special/studentTCdf')

/** sd₂ / sd₁ as a column: every statistic below has one row per ratio and one column per simulated experiment. */
const RATIO = reshape(tensor(LOG_RATIOS.map((l) => 2 ** l)), [LOG_RATIOS.length, 1])
const SHAPE = [LOG_RATIOS.length, REPS]

/** A child of the figure's seeded stream, named by `path`. */
const streamAt = (...path: unknown[]) =>
  call('foundation/random/child', call('foundation/random/stream', SEED), ...path)

/**
 * One group's sample mean and sample variance in every simulated experiment of a row, for a normal group of size n
 * with mean 0 and standard deviation `sd`. They are drawn from their exact sampling laws, x̄ ~ N(0, sd²/n) and
 * (n − 1)s²/sd² ~ χ²ₙ₋₁, independent of each other (Cochran's theorem). Both t statistics depend on a sample only
 * through x̄ and s², so this is the same experiment as drawing the n values, at a cost that does not grow with n.
 *
 * The means are drawn afresh for every cell. The χ² draws are shared across a row's ratios and scaled by sd², because
 * the χ² sampler costs about 20 times a normal draw and the worker evaluates every use of a value in the task tree
 * again. Sharing them correlates the cells of a row only weakly: the simulated Welch map has row means between 0.049
 * and 0.054 and a cell standard deviation of 0.0068, as independent cells at a rate of 0.05 would.
 */
function sampleMoments(n: number, sd: unknown, ...path: unknown[]) {
  const m = call<Tensor>('foundation/random/normals', streamAt('mean', ...path), SHAPE, 0, div(sd, Math.sqrt(n)))
  const chi = call<Tensor>('probability/samplers/chiSquare', streamAt('variance', ...path), n - 1, { shape: [1, REPS] })
  return { m, v: mul(square(sd), div(chi, n - 1)) }
}

/** The share of experiments in each row whose two-sided p-value 2 F_ν(−|t|) is below α. */
const rejectionRate = (t: Task<Tensor>, df: unknown) => mean(less(mul(2, studentTCdf(neg(abs(t)), df)), ALPHA), 1)

/**
 * One row of each map (group 2 of size n₂): both tests on the same simulated experiments.
 * Both groups have mean 0, so every rejection is a false positive.
 */
function row(n2: number, j: number) {
  const g1 = sampleMoments(N1, 1, 'group-1', j)
  const g2 = sampleMoments(n2, RATIO, 'group-2', j)
  const diff = sub(g1.m, g2.m)
  // Welch: each group's own variance, Welch–Satterthwaite degrees of freedom.
  const u = div(g1.v, N1)
  const w = div(g2.v, n2)
  const welchT = div(diff, sqrt(add(u, w)))
  const welchDf = div(square(add(u, w)), add(div(square(u), N1 - 1), div(square(w), n2 - 1)))
  // Pooled: one common variance, n₁ + n₂ − 2 degrees of freedom.
  const pooledVar = div(add(mul(N1 - 1, g1.v), mul(n2 - 1, g2.v)), N1 + n2 - 2)
  const pooledT = div(diff, sqrt(mul(pooledVar, 1 / N1 + 1 / n2)))
  return { pooled: rejectionRate(pooledT, N1 + n2 - 2), welch: rejectionRate(welchT, welchDf) }
}

/** Both maps as one tensor of shape [2, n₂ values, ratios]: the pooled test's map, then Welch's. */
const simulation = () => {
  const rows = N2S.map(row)
  return stack([stack(rows.map((r) => r.pooled)), stack(rows.map((r) => r.welch))])
}

type Maps = { pooled: number[][]; welch: number[][] }

/** Simulated type I error of the pooled and Welch tests over a grid of designs, computed in the compute worker. */
export function TypeIErrorMaps() {
  const run = useComputed<Tensor, Maps | null>(simulation, [], {
    mode: 'worker',
    initial: null,
    then: (r) => {
      const [pooled, welch] = toArray(r) as number[][][]
      return { pooled, welch }
    },
    cancelAfter: 10_000,
  })
  // Both maps share their axes: the same grid of designs.
  const x = useAxis({ label: 'log₂(sd₂ / sd₁)', nice: false })
  const y = useAxis({ label: 'n₂', nice: false })
  // Centre the diverging scale on the nominal α: white is correct, blue too cautious, red too many false positives.
  const range: [number, number] = [0, 2 * ALPHA]
  const map = (z: number[][] | undefined, title: string) => (
    <Plot x={x} y={y} title={title}>
      {z && <Raster x={LOG_RATIOS} y={N2S} z={z} valueLabel="Type I error" scale="diverging" range={range} />}
    </Plot>
  )
  return (
    <Figure
      title="How often each test rejects a true null"
      caption={`Both groups have the same mean, so every rejection is a false positive; a correct test rejects ${
        100 * ALPHA
      }% of the time (the neutral colour). Group 1 has n₁ = ${N1}. Across the x-axis, group 2's standard deviation runs from a quarter to four times group 1's. Red means too many false positives, blue too few. Each cell is ${REPS.toLocaleString(
        'en',
      )} simulated experiments, computed in your browser: each group's sample mean and variance are drawn from their exact normal and χ² sampling laws. A cell's Monte Carlo standard deviation is about 0.007 at a true rate of 0.05.${
        run.error ? ` (Simulation failed: ${run.error}.)` : ''
      }`}
    >
      <Plots cols={2} height={340}>
        {map(run.value?.pooled, "Pooled (Student's) t-test")}
        {map(run.value?.welch, "Welch's t-test")}
      </Plots>
    </Figure>
  )
}
