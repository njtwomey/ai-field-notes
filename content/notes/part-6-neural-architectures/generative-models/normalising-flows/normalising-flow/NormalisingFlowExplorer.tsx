import { useMemo, useState } from 'react'
import { ControlRow, Figure, Plot, Plots, Points, Raster, Readout, Select, formatNumber, useAxis } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { moons, pinwheel, spirals, gaussianRing } from 'aifn-methods/data/synthetic'
import { realNvpRun, type RealNvpCheckpoint, type RealNvpRun } from 'aifn-methods/generative/flows'

const DATASETS = [
  { value: 'moons', label: 'Two moons dataset' },
  { value: 'ring', label: '8-Gaussian ring' },
  { value: 'pinwheel', label: 'Pinwheel dataset' },
  { value: 'spirals', label: 'Two spirals dataset' },
]

export function NormalisingFlowExplorer() {
  const [dataChoice, setDataChoice] = useState<'moons' | 'ring' | 'pinwheel' | 'spirals'>('moons')
  const [layers, setLayers] = useState(4)
  const [steps, setSteps] = useState(400)
  const [layerIndex, setLayerIndex] = useState(0)

  // Run the RealNVP training simulation
  const run: RealNvpRun | null = useMemo(() => {
    const s = stream(`flow-${dataChoice}`)
    let rawData
    if (dataChoice === 'moons') {
      rawData = moons(s, { n: 400, noise: 0.08 })
    } else if (dataChoice === 'ring') {
      rawData = gaussianRing(s, { n: 400 })
    } else if (dataChoice === 'pinwheel') {
      rawData = pinwheel(s, { n: 400, arms: 4, noise: 0.05 })
    } else {
      rawData = spirals(s, { n: 400, arms: 2, noise: 0.04 })
    }

    const gen = realNvpRun(rawData, {
      layers,
      hidden: [24, 24],
      steps,
      stepSize: 0.003,
      batchSize: 128,
      grid: 36,
      samples: 300,
      checkpoints: 8,
      seed: 42,
    })

    let last: RealNvpRun | null = null
    for (const snap of gen) {
      last = snap
    }
    return last
  }, [dataChoice, layers, steps])

  const lastCheckpoint: RealNvpCheckpoint | null = useMemo(() => {
    if (!run || run.checkpoints.length === 0) return null
    return run.checkpoints[run.checkpoints.length - 1]
  }, [run])

  const box = run?.box ?? 3.5
  const gridAxis = useMemo(() => (run ? Array.from(run.gridX) : []), [run])

  // Points transformed after selected coupling layer
  const effectiveLayer = Math.min(layerIndex, layers)
  const layerPoints = useMemo(() => {
    if (!lastCheckpoint || !lastCheckpoint.layers[effectiveLayer]) return { x: [], y: [] }
    const flat = lastCheckpoint.layers[effectiveLayer]
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [lastCheckpoint, effectiveLayer])

  // Generated samples from base Gaussian
  const samplePoints = useMemo(() => {
    if (!lastCheckpoint) return { x: [], y: [] }
    const flat = lastCheckpoint.samples
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [lastCheckpoint])

  // Continuous model density p(x)
  const densityMatrix = useMemo(() => {
    if (!lastCheckpoint || !run) return []
    const g = run.gridX.length
    const flat = lastCheckpoint.density
    const m: number[][] = []
    for (let i = 0; i < g; i++) {
      const row: number[] = []
      for (let j = 0; j < g; j++) {
        row.push(flat[i * g + j])
      }
      m.push(row)
    }
    return m
  }, [lastCheckpoint, run])

  const planeAxisX = useAxis({ label: 'x₁', range: [-box, box] })
  const planeAxisY = useAxis({ label: 'x₂', range: [-box, box] })

  const layerAxisX = useAxis({
    label: effectiveLayer === 0 ? 'Data x₁' : `h₁ (layer ${effectiveLayer})`,
    range: [-box, box],
  })
  const layerAxisY = useAxis({
    label: effectiveLayer === 0 ? 'Data x₂' : `h₂ (layer ${effectiveLayer})`,
    range: [-box, box],
  })

  return (
    <Figure
      title="RealNVP Normalising Flow & Affine Coupling Layers"
      purpose="Explore invertible affine transformations, exact change-of-variables log-likelihood, and multi-modal to Gaussian normalisation."
      caption={
        'RealNVP normalising flow (Dinh et al., 2017). Left: learned continuous density and generated samples x = f(z) from Gaussian noise z ~ N(0, I). Right: the forward transformation of data through successive affine coupling layers towards the base Gaussian (normalisation), where complex multi-modal data is untangled into an isotropic distribution.'
      }
    >
      <ControlRow label="Model configuration">
        <Select
          label="Target dataset"
          value={dataChoice}
          options={DATASETS}
          onChange={(v) => setDataChoice(v as any)}
        />
        <Select
          label="Coupling layers"
          value={String(layers)}
          options={[
            { value: '2', label: '2 layers' },
            { value: '4', label: '4 layers' },
            { value: '6', label: '6 layers' },
          ]}
          onChange={(v) => setLayers(Number(v))}
        />
        <Select
          label="Training steps"
          value={String(steps)}
          options={[
            { value: '200', label: '200 steps' },
            { value: '400', label: '400 steps' },
          ]}
          onChange={(v) => setSteps(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Coupling layer flow (data → base Gaussian)">
        <Select
          label="View layer"
          value={String(effectiveLayer)}
          options={Array.from({ length: layers + 1 }, (_, i) => ({
            value: String(i),
            label: i === 0 ? '0: Data space' : i === layers ? `${i}: Base Gaussian` : `Layer ${i}`,
          }))}
          onChange={(v) => setLayerIndex(Number(v))}
        />
      </ControlRow>

      <Plots>
        <Plot x={planeAxisX} y={planeAxisY} title="Model density & generated samples">
          {densityMatrix.length > 0 && gridAxis.length > 0 && (
            <Raster x={gridAxis} y={gridAxis} z={densityMatrix} scale="sequential" />
          )}
          <Points x={samplePoints.x} y={samplePoints.y} slot={1} size={4} />
        </Plot>

        <Plot
          x={layerAxisX}
          y={layerAxisY}
          title={
            effectiveLayer === layers
              ? 'Base Gaussian space z = f⁻¹(x)'
              : `Representation at layer ${effectiveLayer} of ${layers}`
          }
        >
          <Points x={layerPoints.x} y={layerPoints.y} slot={0} size={4} />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Final NLL"
          value={
            run && run.nll.value.length > 0
              ? formatNumber(Number(run.nll.value[run.nll.value.length - 1].toFixed(3)))
              : '—'
          }
        />
        <Readout label="Coupling layers" value={layers} />
        <Readout label="Conditioner hidden" value="[24, 24]" />
        <Readout
          label="Current representation"
          value={
            effectiveLayer === 0
              ? 'Data input'
              : effectiveLayer === layers
                ? 'Latent N(0, I)'
                : `Coupling layer ${effectiveLayer}`
          }
        />
      </ControlRow>
    </Figure>
  )
}
