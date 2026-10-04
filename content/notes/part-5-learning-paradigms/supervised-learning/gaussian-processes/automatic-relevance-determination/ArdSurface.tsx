import { useMemo } from 'react'
import {
  Button,
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { addDiagonal, cholesky, forward, logDet, gridMaximum } from '../_shared/gp'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 40
const NOISE = 0.2
const LOG_ELL = toFlat(linspace(-1, 2, 36))
const DEPTH = 40

type Target = 'x1' | 'both'
const TARGETS = [
  { value: 'x1' as const, label: 'x₁ only' },
  { value: 'both' as const, label: 'x₁ and x₂' },
]

function makeData(seed: number, target: Target) {
  const g = stream(seed)
  const x = Array.from({ length: N }, () => [-2 + 4 * uniform(g), -2 + 4 * uniform(g)])
  const y = x.map(([a, b]) => Math.sin(1.5 * a) + (target === 'both' ? 0.8 * Math.cos(1.5 * b) : 0) + NOISE * normal(g))
  return { x, y }
}

/** ln p(y) for the ARD squared exponential kernel with σ_f = 1 and the true noise level. */
function logEvidence(x: number[][], y: number[], ell1: number, ell2: number): number {
  const K = x.map(([a1, a2]) =>
    x.map(([b1, b2]) => Math.exp(-0.5 * (((a1 - b1) / ell1) ** 2 + ((a2 - b2) / ell2) ** 2))),
  )
  const l = cholesky(addDiagonal(K, NOISE * NOISE))
  const z = forward(l, y)
  return -0.5 * z.reduce((s, v) => s + v * v, 0) - 0.5 * logDet(l) - 0.5 * N * Math.log(2 * Math.PI)
}

/** The marginal likelihood over the two ARD length-scales of a GP on two inputs. */
export function ArdSurface() {
  const fmtPow = (v: number) => formatNumber(10 ** v)
  const state = useFigureState({
    target: choice<Target>(TARGETS, 'x1', { label: 'target depends on' }),
    log1: slider(-1, 2, 0, { step: 0.01, label: 'length-scale ℓ₁', format: fmtPow }),
    log2: slider(-1, 2, 0, { step: 0.01, label: 'length-scale ℓ₂', format: fmtPow }),
    seed: int(2, { min: 1, max: 20, label: 'data seed' }),
  })
  const target = state.target
  const seed = { value: state.seed }
  const log1 = { value: state.log1, set: (v: number) => state.set('log1', v) }
  const log2 = { value: state.log2, set: (v: number) => state.set('log2', v) }
  const data = useMemo(() => makeData(seed.value, target), [seed.value, target])

  const surface = useMemo(() => {
    const z = LOG_ELL.map((l2) => LOG_ELL.map((l1) => logEvidence(data.x, data.y, 10 ** l1, 10 ** l2)))
    const best = gridMaximum(z)
    const range: [number, number] = [best.value - DEPTH, best.value]
    return { z: z.map((row) => row.map((v) => Math.max(v, range[0]))), best, range }
  }, [data])

  const here = useMemo(
    () => logEvidence(data.x, data.y, 10 ** log1.value, 10 ** log2.value),
    [data, log1.value, log2.value],
  )
  const bestEll1 = LOG_ELL[surface.best.j]
  const bestEll2 = LOG_ELL[surface.best.i]

  const xAxis = useAxis({ label: 'log₁₀ ℓ₁', hold: 'union' })
  const yAxis = useAxis({ label: 'log₁₀ ℓ₂', hold: 'union' })
  return (
    <Figure
      title="Switching off an irrelevant input"
      state={state}
      caption="Forty points with two inputs in [−2, 2]². The target is sin 1.5x₁ plus noise, or, with the second choice, also depends on x₂. The heatmap is the log marginal likelihood of a GP with one length-scale per input, over log₁₀ ℓ₁ (horizontal) and log₁₀ ℓ₂ (vertical). Drag the point to set both. When the target ignores x₂, the surface keeps rising as ℓ₂ grows and flattens into a plateau at the top: the best model makes the kernel constant along x₂. When x₂ matters, the maximum moves to a finite ℓ₂."
      controls={
        <>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              log1.set(bestEll1)
              log2.set(bestEll2)
            }}
          >
            Go to the maximum
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="ln p(y | ℓ₁, ℓ₂)" value={formatNumber(here)} />
          <Readout label="grid maximum ℓ₁, ℓ₂" value={`${fmtPow(bestEll1)}, ${fmtPow(bestEll2)}`} />
          <Readout label="relevance 1/ℓ₁, 1/ℓ₂" value={`${fmtPow(-log1.value)}, ${fmtPow(-log2.value)}`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={380}>
          <Raster
            x={LOG_ELL}
            y={LOG_ELL}
            z={surface.z}
            scale={'sequential'}
            range={surface.range}
            valueLabel={'ln p(y | ℓ₁, ℓ₂)'}
          />
          <Handle {...state.handle(['log1', 'log2'], { label: '(ℓ₁, ℓ₂)' })} />
        </Plot>
      </div>
    </Figure>
  )
}
