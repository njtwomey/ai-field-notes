import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type HeatmapOverlay,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { decision, rbfKernel, trainSvm, type Point } from '../_shared/svm'

type Shape = 'circles' | 'moons' | 'xor'
const SHAPES = [
  { value: 'circles' as const, label: 'rings' },
  { value: 'moons' as const, label: 'moons' },
  { value: 'xor' as const, label: 'XOR' },
]
const N = 60
const AXIS = linspace(-3, 3, 49)
const RANGE: [number, number] = [-2, 2]

function makeData(shape: Shape, seed: number, noise: number) {
  const g = rng(seed)
  const x: Point[] = []
  const y: number[] = []
  for (let i = 0; i < N; i++) {
    const label = i % 2 === 0 ? 1 : -1
    let p: Point
    if (shape === 'circles') {
      const r = label > 0 ? 0.8 : 2
      const t = 2 * Math.PI * g.uniform()
      p = [r * Math.cos(t), r * Math.sin(t)]
    } else if (shape === 'moons') {
      const t = Math.PI * g.uniform()
      p = label > 0 ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25] : [0.5 - Math.cos(t), 0.25 - Math.sin(t)]
      p = [1.4 * p[0], 1.4 * p[1]]
    } else {
      const a = g.uniform() < 0.5 ? -1 : 1
      const b = label > 0 ? a : -a
      p = [a * (0.4 + 1.6 * g.uniform()), b * (0.4 + 1.6 * g.uniform())]
    }
    x.push([p[0] + noise * g.normal(), p[1] + noise * g.normal()])
    y.push(label)
  }
  return { x, y }
}

/** An RBF-kernel SVM: the decision function over the plane, its sign giving the class, and the support vectors. */
export function KernelSvm() {
  const [shape, setShape] = useState<Shape>('circles')
  const logEll = useParam(-0.2, { min: -1, max: 0.6, step: 0.02 })
  const logC = useParam(1, { min: -1, max: 3, step: 0.05 })
  const noise = useParam(0.2, { min: 0, max: 0.6, step: 0.02 })
  const seed = useParam(2, { min: 1, max: 20, step: 1 })
  const ell = 10 ** logEll.value
  const C = 10 ** logC.value

  const data = useMemo(() => makeData(shape, seed.value, noise.value), [shape, seed.value, noise.value])
  const r = useMemo(() => {
    const kernel = rbfKernel(ell)
    const fit = trainSvm(data.x, data.y, C, kernel)
    const z = AXIS.map((v) => AXIS.map((u) => Math.max(-2, Math.min(2, decision(fit, data.x, data.y, kernel, [u, v])))))
    const sv = fit.alpha.map((a, i) => (a > 1e-6 ? i : -1)).filter((i) => i >= 0)
    const errors = data.x.filter((p, i) => data.y[i] * decision(fit, data.x, data.y, kernel, p) < 0).length
    return { z, sv, errors, atBound: sv.filter((i) => fit.alpha[i] > C * (1 - 1e-6)).length }
  }, [data, ell, C])

  const overlay: HeatmapOverlay[] = [
    {
      name: 'points',
      type: 'scatter',
      x: data.x.map((p) => p[0]),
      y: data.x.map((p) => p[1]),
      group: data.y.map((v) => (v > 0 ? 1 : 0)),
      groupNames: ['class −1', 'class +1'],
    },
    {
      name: 'support vectors',
      type: 'scatter',
      x: r.sv.map((i) => data.x[i][0]),
      y: r.sv.map((i) => data.x[i][1]),
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="A kernel SVM with the Gaussian kernel"
      caption="The colour is the decision function f(x) = Σ αᵢ yᵢ k(xᵢ, x) + b, clipped to [−2, 2]: red where the SVM predicts class +1, blue for class −1, and the neutral band around the boundary f = 0. Diamonds mark the support vectors. A short length-scale ℓ lets every support vector carve its own island, and with a large C the boundary wraps around individual points. A long length-scale makes the kernel nearly linear over the data and the boundary nearly straight, which cannot separate rings or XOR."
      controls={
        <>
          <ParamChoice label="data" value={shape} onChange={setShape} options={SHAPES} />
          <ParamSlider label="kernel length-scale ℓ" param={logEll} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="C" param={logC} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="noise" param={noise} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="support vectors" value={`${r.sv.length} of ${N} (${r.atBound} at C)`} />
          <Readout label="training errors" value={String(r.errors)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={r.z}
          scale="diverging"
          range={RANGE}
          overlay={overlay}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="f(x)"
          height={420}
        />
      </div>
    </Interactive>
  )
}
