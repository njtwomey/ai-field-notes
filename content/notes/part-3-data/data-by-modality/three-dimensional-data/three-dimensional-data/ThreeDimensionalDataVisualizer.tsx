import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ThreeDimensionalDataVisualizer() {
  const state = useFigureState({
    mode: choice<'voxel_scaling' | 'lidar_decay' | 'volume_rendering'>(
      [
        {
          value: 'voxel_scaling',
          label: '3D Representations: Dense Voxel R³ Memory Blowout vs Sparse Point Sets (Qi 2017)',
        },
        {
          value: 'lidar_decay',
          label: 'Point Clouds: Range Distance vs Surface Point Density Decay ~ 1/d² (PointNet)',
        },
        {
          value: 'volume_rendering',
          label: 'Implicit NeRF: Transmittance T(t) & Accumulated Opacity vs Depth t (Mildenhall 2020)',
        },
      ],
      'voxel_scaling',
      { label: '3d modality analysis' },
    ),
    param1: slider(16, 256, 64, {
      step: 16,
      label: 'voxel grid resolution R / beam divergence / scene density sigma',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'sparsity factor (% occupied) / sensor scan rate / ray sample count',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'voxel_scaling') {
      const resR = param1 // 16 to 256
      const sparsityPct = param2 * 0.5 // 0.5% to 5.0% surface occupancy

      // Resolution from 16 to 256
      const resolutions = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256]
      const denseMemoryMB: number[] = []
      const sparseMemoryMB: number[] = []

      for (const r of resolutions) {
        // Dense 32-bit float: R^3 * 4 bytes / 10^6
        const dense = (Math.pow(r, 3) * 4) / (1024 * 1024)
        // Sparse: only occupied voxels stored in hash map (16 bytes per active cell)
        const sparse = (Math.pow(r, 3) * (sparsityPct / 100) * 16) / (1024 * 1024)

        denseMemoryMB.push(Math.min(2048, dense))
        sparseMemoryMB.push(sparse)
      }

      const activeDense = (Math.pow(resR, 3) * 4) / (1024 * 1024)
      const activeSparse = (Math.pow(resR, 3) * (sparsityPct / 100) * 16) / (1024 * 1024)

      return {
        x: resolutions,
        y1: denseMemoryMB,
        y2: sparseMemoryMB,
        stat1: `Resolution: ${resR}³ grid`,
        stat2: `Dense Voxel: ${activeDense >= 1024 ? (activeDense / 1024).toFixed(2) + ' GB' : activeDense.toFixed(1) + ' MB'}`,
        stat3: `Sparse Active: ${activeSparse.toFixed(1)} MB`,
        insight:
          'Dense voxel tensors scale cubically O(R³), making sparse point clouds or implicit fields mandatory for fine detail',
        axisXLabel: 'Voxel Grid Resolution R (cells per axis)',
        axisYLabel: 'GPU Memory Footprint (MB)',
        xRange: [10, 265] as [number, number],
        yRange: [0, 1100] as [number, number],
      }
    } else if (mode === 'lidar_decay') {
      // LiDAR point cloud surface density decay rho ~ 1 / d^2
      const distances = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      const densities: number[] = []
      const relativeSpacing: number[] = []

      for (const d of distances) {
        // Angular resolution delta_theta = 0.1 deg = 0.00174 rad
        // Point spacing delta_s = d * delta_theta
        // Surface density rho = 1 / (d * delta_theta)^2
        const spacingCm = d * 0.174
        const densityPerM2 = 10000 / (spacingCm * spacingCm)

        densities.push(Math.min(500, densityPerM2))
        relativeSpacing.push(spacingCm)
      }

      const activeDist = Math.max(5, param1 * 0.3)
      const activeSpacing = activeDist * 0.174
      const activeDensity = 10000 / (activeSpacing * activeSpacing)

      return {
        x: distances,
        y1: densities,
        y2: relativeSpacing.map((s) => s * 5),
        stat1: `Target Distance d = ${activeDist.toFixed(0)} m`,
        stat2: `Density: ${activeDensity.toFixed(0)} pts/m²`,
        stat3: `Point Spacing: ${activeSpacing.toFixed(1)} cm`,
        insight: 'Non-uniform point density: distant objects have 100x fewer points than foreground objects',
        axisXLabel: 'Sensor Distance d (m)',
        axisYLabel: 'Surface Density (pts/m²) & Spacing (x5 cm)',
        xRange: [0, 105] as [number, number],
        yRange: [0, 520] as [number, number],
      }
    } else {
      // NeRF Volume rendering ray marching: depth t from 1.0 to 5.0
      const depths = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
      const transmittanceT: number[] = []
      const accumulatedOpacity: number[] = []

      const sigma = param2 * 0.4 // density scale

      let accumulatedSigma = 0
      for (const t of depths) {
        // Object surface centered around t = 3.0
        const localSigma = sigma * Math.exp(-Math.pow(t - 3.0, 2) / 0.3)
        accumulatedSigma += localSigma * 0.4
        const trans = Math.exp(-accumulatedSigma)

        transmittanceT.push(trans * 100)
        accumulatedOpacity.push((1 - trans) * 100)
      }

      return {
        x: depths,
        y1: transmittanceT,
        y2: accumulatedOpacity,
        stat1: 'Volume Rendering Integral',
        stat2: 'T(t) = exp(-∫ σ ds)',
        stat3: 'Continuous Ray Marching',
        insight:
          'Transmittance falls to zero as the ray penetrates solid geometry, terminating background contribution',
        axisXLabel: 'Ray Marching Distance t',
        axisYLabel: 'Transmittance T(t) & Accumulated Opacity (%)',
        xRange: [0.8, 5.2] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Three-Dimensional Data: Representations, Memory Footprint, and Geometric Symmetries"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Representation Mode" value={results.stat1} />
          <Readout label="Primary Metric / Scale" value={results.stat2} />
          <Readout label="Efficiency / Symmetry" value={results.stat3} />
          <Readout label="Geometric Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Computational and Discretization Profile Across 3D Formats">
        <Curve
          name={
            mode === 'voxel_scaling'
              ? 'Dense 3D Voxel Grid Tensor Memory (MB)'
              : mode === 'lidar_decay'
                ? 'LiDAR Surface Point Density (pts/m²)'
                : 'Ray Transmittance T(t) (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'voxel_scaling'
              ? 'Sparse Submanifold Hash Grid Memory (MB)'
              : mode === 'lidar_decay'
                ? 'LiDAR Point-to-Point Spacing (x5 cm)'
                : 'Accumulated Surface Opacity 1 - T(t) (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
