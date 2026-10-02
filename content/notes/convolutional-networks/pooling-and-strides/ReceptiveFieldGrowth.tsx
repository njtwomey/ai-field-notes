import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, useParam, type XYSeries } from 'aifn-render'

type Layer = { k: number; s: number; pool: boolean }

/** Stages of `perStage` k × k stride-1 convolutions, each stage followed by a 2 × 2 stride-2 pool if `pool`. */
function network(stages: number, perStage: number, k: number, pool: boolean): Layer[] {
  const layers: Layer[] = []
  for (let st = 0; st < stages; st++) {
    for (let c = 0; c < perStage; c++) layers.push({ k, s: 1, pool: false })
    if (pool) layers.push({ k: 2, s: 2, pool: true })
  }
  return layers
}

/** r_l = r_{l-1} + (k_l - 1) j_{l-1}, j_l = j_{l-1} s_l, starting from r_0 = j_0 = 1. */
function receptiveFields(layers: Layer[]): { r: number[]; j: number[] } {
  const r = [1]
  const j = [1]
  for (const { k, s } of layers) {
    r.push(r[r.length - 1] + (k - 1) * j[j.length - 1])
    j.push(j[j.length - 1] * s)
  }
  return { r, j }
}

/** Receptive-field size after each layer, with and without downsampling between stages. */
export function ReceptiveFieldGrowth() {
  const stages = useParam(5, { min: 1, max: 5, step: 1 })
  const perStage = useParam(2, { min: 1, max: 3, step: 1 })
  const [kernel, setKernel] = useState<'3' | '5'>('3')
  const k = Number(kernel)

  const { series, pooled, plain } = useMemo(() => {
    const withPool = network(stages.value, perStage.value, k, true)
    const pooled = receptiveFields(withPool)
    // The same convolutions with the pooling layers removed.
    const withoutPool = network(stages.value, perStage.value, k, false)
    const plain = receptiveFields(withoutPool)
    const convIndex = (layers: Layer[]) => {
      // Plot against the number of convolution layers so both networks share the x axis.
      const xs = [0]
      let n = 0
      for (const l of layers) xs.push(l.pool ? n : ++n)
      return xs
    }
    const series: XYSeries[] = [
      { name: 'with 2 × 2 pooling after each stage', type: 'line', x: convIndex(withPool), y: pooled.r, slot: 0 },
      { name: 'no pooling', type: 'line', x: convIndex(withoutPool), y: plain.r, slot: 1 },
    ]
    return { series, pooled, plain }
  }, [stages.value, perStage.value, k])

  const last = (a: number[]) => a[a.length - 1]

  return (
    <Interactive
      title="Receptive-field growth"
      caption="Each convolution adds (k − 1) × jump to the receptive field, where the jump is the product of all earlier strides. Without pooling the field grows linearly with depth. With a stride-2 pool after each stage the jump doubles per stage, so later layers add more pixels per layer and the growth is geometric. The vertical steps of the blue curve are the pooling layers."
      controls={
        <>
          <ParamSlider label="stages" param={stages} format={(v) => String(v)} withArrows />
          <ParamSlider label="convolutions per stage" param={perStage} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="kernel k"
            value={kernel}
            onChange={setKernel}
            options={[
              { value: '3', label: '3 × 3' },
              { value: '5', label: '5 × 5' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="receptive field, pooled" value={`${last(pooled.r)} × ${last(pooled.r)}`} />
          <Readout label="receptive field, no pooling" value={`${last(plain.r)} × ${last(plain.r)}`} />
          <Readout label="final jump" value={last(pooled.j)} />
        </>
      }
    >
      <XYChart series={series} xLabel="convolution layers" yLabel="receptive field (pixels per side)" height={300} />
    </Interactive>
  )
}
