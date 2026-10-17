import { useMemo, useState } from 'react'
import {
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
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { moons } from 'aifn-methods/data'
import { pointAffinity } from 'aifn-compute/graph/propagation'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { child, stream } from 'aifn-compute/foundation/random'
import { matmul, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { laplacianRegularisedLeastSquares } from 'aifn-methods/learning/semi-supervised'
import { laplacian } from 'aifn-compute/graph/matrices'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'

const N = 120
const N_TEST = 200
const GRAPH_SIGMA = 0.2
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number, number] = [-1, 1.5]
const NX = 48
const NY = 36
const GX = Array.from({ length: NX }, (_, j) => X_RANGE[0] + ((j + 0.5) * (X_RANGE[1] - X_RANGE[0])) / NX)
const GY = Array.from({ length: NY }, (_, i) => Y_RANGE[0] + ((i + 0.5) * (Y_RANGE[1] - Y_RANGE[0])) / NY)
const GRID = tensor(GY.flatMap((gy) => GX.map((gx) => [gx, gy])))

/** The share of rows whose sign of f matches the class (class 1 is f > 0). */
const signAccuracy = (f: number[], y: number[], rows: readonly number[]) =>
  rows.filter((i) => (f[i] > 0 ? 1 : 0) === y[i]).length / rows.length

/**
 * Laplacian regularised least squares on two moons with one label per class, beside the same kernel model without the
 * graph term (kernel ridge regression), with predictions on the whole plane and on a fresh test set.
 */
export function LaplacianBoundary() {
  const fig = useFigureState({
    gammaI: float(1000, {
      gt: 0,
      le: 1e6,
      scale: 'log10',
      label: 'intrinsic weight γ_I',
      suggestions: [10, 100, 1000, 1e4],
    }),
    gammaA: float(0.001, {
      gt: 0,
      le: 10,
      scale: 'log10',
      label: 'ambient weight γ_A',
      suggestions: [1e-4, 1e-3, 1e-2, 0.1],
    }),
    lengthscale: float(0.3, {
      gt: 0,
      le: 5,
      step: 0.05,
      label: 'kernel lengthscale ℓ',
      suggestions: [0.2, 0.3, 0.5, 1],
    }),
    k: int(8, { min: 2, max: 40, label: 'graph neighbours k', suggestions: [4, 8, 16] }),
    perClass: int(1, { min: 1, max: 20, label: 'labels per class', suggestions: [1, 2, 5] }),
    seed: int(3, { min: 1, max: 999, label: 'data seed' }),
  })
  const { gammaI, gammaA, lengthscale, k, perClass, seed } = fig
  const [probe, setProbe] = useState<[number, number]>([0.5, 0.25])

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.08, spacing: 'random' })
    const test = moons(child(stream(seed), 'test'), { n: N_TEST, noise: 0.08, spacing: 'random' })
    const y = toFlat(d.y)
    const xy = toFlat(d.x)
    return {
      x: d.x,
      y,
      test: test.x,
      yTest: toFlat(test.y),
      px: y.map((_, i) => xy[2 * i]),
      py: y.map((_, i) => xy[2 * i + 1]),
    }
  }, [seed])
  const split = useMemo(
    () => semiSupervisedSplit(child(stream(seed), 'split'), data.y, perClass),
    [data, seed, perClass],
  )
  const graphLaplacian = useMemo(
    () => laplacian(pointAffinity(data.x, { gamma: 1 / (2 * GRAPH_SIGMA ** 2), neighbours: k })),
    [data, k],
  )

  const fits = useComputed(() => {
    const kernel = rbf({ lengthscale })
    const K = gram(kernel, data.x)
    const targets = data.y.map((v) => (v === 1 ? 1 : -1))
    const allRows = data.yTest.map((_, i) => i)
    return [0, gammaI].map((g) => {
      const m = laplacianRegularisedLeastSquares(K, graphLaplacian, targets, split.labelled, { gammaA, gammaI: g })
      const field = toFlat(matmul(gram(kernel, GRID, data.x), m.alpha))
      const test = toFlat(matmul(gram(kernel, data.test, data.x), m.alpha))
      return {
        alpha: m.alpha,
        z: GY.map((_, i) => field.slice(i * NX, (i + 1) * NX)),
        unlabelled: signAccuracy(toFlat(m.fitted), data.y, split.unlabelled),
        test: signAccuracy(test, data.yTest, allRows),
      }
    })
  }, [data, split, graphLaplacian, gammaA, gammaI, lengthscale])

  const atProbe = (alpha: (typeof fits.value)[number]['alpha']) =>
    toFlat(matmul(gram(rbf({ lengthscale }), tensor([probe]), data.x), alpha))[0]

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const U = split.unlabelled
  const L = split.labelled
  const [rls, lap] = fits.value

  const panel = (z: number[][], title: string, withProbe: boolean) => (
    <Plot x={xa} y={ya} title={title}>
      <Raster
        x={GX}
        y={GY}
        z={z}
        scale="diverging"
        range={[-1, 1]}
        valueLabel="f(x)"
        boundary
        fillOpacity={0.55}
        stale={fits.stale}
      />
      <Points name="unlabelled" x={U.map((i) => data.px[i])} y={U.map((i) => data.py[i])} muted thin />
      {[0, 1].map((c) => {
        const rows = L.filter((i) => data.y[i] === c)
        return rows.length > 0 ? (
          <Points
            key={c}
            name={`labelled class ${c}`}
            x={rows.map((i) => data.px[i])}
            y={rows.map((i) => data.py[i])}
            slot={c}
            size={11}
          />
        ) : null
      })}
      {withProbe && <Handle kind="point" at={probe} label="new point" onDrag={setProbe} />}
    </Plot>
  )

  return (
    <Figure
      title="The Laplacian term bends the boundary along the moons"
      state={fig}
      caption={`${N} points on two moons, ${perClass} labelled per class (large marks). Both panels fit f(x) = Σ αᵢ k(xᵢ, x) with a Gaussian kernel of lengthscale ℓ; the shading is f and the ink line its zero level, the decision boundary. Left: γ_I = 0, which is kernel ridge regression on the labelled points alone; its boundary is set by where the labels happen to lie. Right: the same model with the penalty (γ_I / n²) fᵀLf on a ${k}-nearest-neighbour graph over all ${N} points (Gaussian edge weights, σ = ${GRAPH_SIGMA}); the boundary moves into the gap between the moons. The test accuracy is on ${N_TEST} fresh points that were not in the graph, so the fitted function predicts new inputs. Drag the marker to read f at any new point.`}
      readouts={{
        'γ_I = 0 (kernel ridge)': (
          <>
            <Readout label="accuracy, unlabelled" value={formatNumber(rls.unlabelled)} />
            <Readout label="accuracy, test" value={formatNumber(rls.test)} />
            <Readout label="f(new point)" value={formatNumber(atProbe(rls.alpha))} />
          </>
        ),
        'with the Laplacian term': (
          <>
            <Readout label="accuracy, unlabelled" value={formatNumber(lap.unlabelled)} />
            <Readout label="accuracy, test" value={formatNumber(lap.test)} />
            <Readout label="f(new point)" value={formatNumber(atProbe(lap.alpha))} />
          </>
        ),
      }}
    >
      <Plots cols={2}>
        {panel(rls.z, 'kernel ridge regression (γ_I = 0)', false)}
        {panel(lap.z, `Laplacian RLS (γ_I = ${formatNumber(gammaI)})`, true)}
      </Plots>
    </Figure>
  )
}
