import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ImagePreprocessingVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'aspect_ratio' | 'resampling_aliasing' | 'tiling_coverage'>(
      [
        { value: 'aspect_ratio', label: 'Aspect Ratio: Anisotropic Squish vs Center Crop vs Letterbox Padding' },
        {
          value: 'resampling_aliasing',
          label: 'Downsampling Resampling: Aliased Nearest vs Anti-Aliased Area (Parmar 2022)',
        },
        { value: 'tiling_coverage', label: 'Gigapixel Tiling: Patch Overlap vs Boundary Seam Redundancy' },
      ],
      'aspect_ratio',
      { label: 'preprocessing operation' },
    ),
    downsampleScale: slider(2, 16, 4, {
      step: 2,
      label: 'downsampling scale factor K (e.g. 4x = 1024 to 256)',
    }),
    overlapPercent: slider(0, 50, 25, {
      step: 5,
      label: 'tiling patch overlap fraction (%)',
    }),
  })

  const { visualizerMode, downsampleScale, overlapPercent } = state

  const results = useMemo(() => {
    if (visualizerMode === 'aspect_ratio') {
      // 6 original aspect ratios: 1:1, 4:3, 16:9, 21:9, 3:2, 9:16 (vertical)
      const ratios = [1.0, 1.33, 1.78, 2.33, 1.5, 0.56]
      // Target square aspect ratio: 1.0 (224 x 224)

      // Geometric distortion for anisotropic stretch: |ratio - 1.0| / ratio * 100
      const stretchDistortion = ratios.map((r) => (Math.abs(r - 1.0) / Math.max(r, 1.0)) * 100)
      // Cropped pixel loss for center-crop:
      const cropLoss = ratios.map((r) => (r >= 1.0 ? (1 - 1 / r) * 100 : (1 - r) * 100))

      const targetIdx = 2 // 16:9
      const currDistortion = stretchDistortion[targetIdx]
      const currCrop = cropLoss[targetIdx]

      return {
        x: ratios.map((_, i) => i + 1),
        y: stretchDistortion,
        baseline: cropLoss,
        stat1: `Target: 224x224 (1:1)`,
        stat2: `16:9 stretch distortion: ${currDistortion.toFixed(1)}%`,
        stat3: `16:9 center crop lost: ${currCrop.toFixed(1)}%`,
        warning:
          'Anisotropic stretching distorts circular objects into ellipses; letterboxing pads borders without cropping or distortion',
        axisXLabel: 'Original Image Aspect Ratio (1=1:1, 2=4:3, 3=16:9, 4=21:9, 5=3:2, 6=9:16)',
        axisYLabel: 'Distortion / Area Loss Penalty (%)',
        xRange: [0.5, 6.5] as [number, number],
        yRange: [0, 65] as [number, number],
      }
    } else if (visualizerMode === 'resampling_aliasing') {
      // Scale factors K from 2 to 16
      const scales = [2, 4, 6, 8, 10, 12, 14, 16]
      // Aliased nearest neighbor artifact energy (% high-frequency noise power)
      const aliasedArtifact = scales.map((k) => Math.min(95, 12 + k * 5.2))
      // Anti-aliased area / Lanczos filter artifact power (stays low and flat)
      const cleanFilter = scales.map(() => 5.0)

      const currAliased = Math.min(95, 12 + downsampleScale * 5.2)

      return {
        x: scales,
        y: aliasedArtifact,
        baseline: cleanFilter,
        stat1: `Scale factor K = ${downsampleScale}x`,
        stat2: `Aliased artifact index: ${currAliased.toFixed(1)}%`,
        stat3: 'Clean anti-aliased: < 5%',
        warning:
          downsampleScale >= 8
            ? 'Severe downsampling aliasing: Unfiltered interpolation generates high-frequency moiré patterns and jagged edges'
            : 'Anti-aliased box/area downsampling integrates pixel areas, eliminating moiré artifacts',
        axisXLabel: 'Downsampling Scale Factor K',
        axisYLabel: 'Aliasing Distortion & Moiré Energy (%)',
        xRange: [1.5, 16.5] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else {
      // Large 8192 x 8192 Whole Slide Image tiled into 512 x 512 patches
      const overlaps = [0, 10, 20, 25, 30, 40, 50]
      // Base non-overlapping tiles = (8192 / 512)^2 = 16^2 = 256 tiles
      // With overlap p, stride S = 512 * (1 - p / 100)
      // Tile count = (8192 / S)^2
      const tileCounts = overlaps.map((p) => {
        const stride = 512 * (1 - p / 100)
        return Math.round(Math.pow(8192 / stride, 2))
      })
      const nonOverlappingBaseline = overlaps.map(() => 256)

      const currStride = 512 * (1 - overlapPercent / 100)
      const currTiles = Math.round(Math.pow(8192 / currStride, 2))

      return {
        x: overlaps,
        y: tileCounts,
        baseline: nonOverlappingBaseline,
        stat1: `Tile overlap: ${overlapPercent}%`,
        stat2: `Patch stride: ${currStride.toFixed(0)} px`,
        stat3: `Total tiles: ${currTiles.toLocaleString()}`,
        warning:
          overlapPercent >= 40
            ? 'High tile inflation: 50% overlap quadruples tile count (1,024 patches), providing dense seam coverage for object detection'
            : 'Controlled tiling: Moderate overlap captures cross-boundary structures without computational exhaustion',
        axisXLabel: 'Patch Border Overlap Fraction (%)',
        axisYLabel: 'Extracted Tile Patch Count',
        xRange: [-2, 52] as [number, number],
        yRange: [200, 1100] as [number, number],
      }
    }
  }, [visualizerMode, downsampleScale, overlapPercent])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Image Preprocessing: Aspect Ratio Handling, Resampling Aliasing, and Gigapixel Tiling"
      state={state}
      hoverReadout={false}
      caption="Essential image preparation transforms: The trade-off between geometric stretching distortion and cropping loss, high-frequency aliasing during unfiltered downsampling, and tile inflation in gigapixel patching."
      readouts={
        <>
          <Readout label="Preprocessing Step" value={results.stat1} />
          <Readout label="Distortion Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'aspect_ratio'
                ? 'Crop Loss'
                : visualizerMode === 'resampling_aliasing'
                  ? 'Anti-Aliased'
                  : 'Total Patches'
            }
            value={results.stat3}
            color={visualizerMode === 'resampling_aliasing' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Pipeline Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Image Transformation Profile">
        <Curve
          name={
            visualizerMode === 'aspect_ratio'
              ? 'Center-Crop Discarded Information Ratio (%)'
              : visualizerMode === 'resampling_aliasing'
                ? 'Anti-Aliased Filter Benchmark (<5% Noise)'
                : 'Non-Overlapping Tile Baseline (256 Patches)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'aspect_ratio'
              ? 'Anisotropic Squish Geometric Distortion (%)'
              : visualizerMode === 'resampling_aliasing'
                ? 'Aliased Nearest Resampling Noise Energy (%)'
                : 'Extracted Patches Under Stride Overlap'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Config" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
