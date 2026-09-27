import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type Handle,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { addDiagonal, cholesky, forward, logDet, gridMaximum } from '../_shared/gp'

const N = 40
const NOISE = 0.2
const LOG_ELL = linspace(-1, 2, 36)
const DEPTH = 40

type Target = 'x1' | 'both'
const TARGETS = [
  { value: 'x1' as const, label: 'x₁ only' },
  { value: 'both' as const, label: 'x₁ and x₂' },
]

function makeData(seed: number, target: Target) {
  const g = rng(seed)
  const x = Array.from({ length: N }, () => [-2 + 4 * g.uniform(), -2 + 4 * g.uniform()])
  const y = x.map(
    ([a, b]) => Math.sin(1.5 * a) + (target === 'both' ? 0.8 * Math.cos(1.5 * b) : 0) + NOISE * g.normal(),
  )
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
  const [target, setTarget] = useState<Target>('x1')
  const seed = useParam(2, { min: 1, max: 20, step: 1 })
  const log1 = useParam(0, { min: -1, max: 2, step: 0.01 })
  const log2 = useParam(0, { min: -1, max: 2, step: 0.01 })
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
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [log1.value, log2.value],
      label: '(ℓ₁, ℓ₂)',
      onDrag: ([a, b]) => {
        log1.set(a)
        log2.set(b)
      },
    },
  ]
  const fmtPow = (v: number) => formatNumber(10 ** v)
  const bestEll1 = LOG_ELL[surface.best.j]
  const bestEll2 = LOG_ELL[surface.best.i]

  return (
    <Interactive
      title="Switching off an irrelevant input"
      caption="Forty points with two inputs in [−2, 2]². The target is sin 1.5x₁ plus noise, or, with the second choice, also depends on x₂. The heatmap is the log marginal likelihood of a GP with one length-scale per input, over log₁₀ ℓ₁ (horizontal) and log₁₀ ℓ₂ (vertical). Drag the point to set both. When the target ignores x₂, the surface keeps rising as ℓ₂ grows and flattens into a plateau at the top: the best model makes the kernel constant along x₂. When x₂ matters, the maximum moves to a finite ℓ₂."
      controls={
        <>
          <ParamChoice label="target depends on" value={target} onChange={setTarget} options={TARGETS} />
          <ParamSlider label="length-scale ℓ₁" param={log1} format={fmtPow} />
          <ParamSlider label="length-scale ℓ₂" param={log2} format={fmtPow} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} />
          <ParamButton
            onClick={() => {
              log1.set(bestEll1)
              log2.set(bestEll2)
            }}
          >
            Go to the maximum
          </ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="ln p(y | ℓ₁, ℓ₂)" value={formatNumber(here)} />
          <Readout label="grid maximum ℓ₁, ℓ₂" value={`${fmtPow(bestEll1)}, ${fmtPow(bestEll2)}`} />
          <Readout label="relevance 1/ℓ₁, 1/ℓ₂" value={`${fmtPow(-log1.value)}, ${fmtPow(-log2.value)}`} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Heatmap
          x={LOG_ELL}
          y={LOG_ELL}
          z={surface.z}
          range={surface.range}
          scale="sequential"
          xLabel="log₁₀ ℓ₁"
          yLabel="log₁₀ ℓ₂"
          valueLabel="ln p(y | ℓ₁, ℓ₂)"
          handles={handles}
          height={380}
        />
      </div>
    </Interactive>
  )
}
