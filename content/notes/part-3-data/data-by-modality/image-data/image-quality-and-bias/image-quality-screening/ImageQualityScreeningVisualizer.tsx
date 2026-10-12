import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ImageQualityScreeningVisualizer() {
  const state = useFigureState({
    screeningMetric: choice<'laplacian_blur' | 'exposure_clipping' | 'brisque_compression' | 'perceptual_hash'>(
      [
        {
          value: 'laplacian_blur',
          label: 'Defocus Blur: Discrete Laplacian Variance Thresholding (Pech-Pacheco 2000)',
        },
        {
          value: 'exposure_clipping',
          label: 'Sensor Exposure: Dynamic Range Saturation and Truncation Rates',
        },
        {
          value: 'brisque_compression',
          label: 'Compression Artifacts: No-Reference BRISQUE NSS Distortion (Mittal 2012)',
        },
        {
          value: 'perceptual_hash',
          label: 'Near-Duplicate Detection: 64-bit DCT pHash Hamming Distance Clusters',
        },
      ],
      'laplacian_blur',
      { label: 'screening dimension' },
    ),
    threshold: slider(20, 500, 120, {
      step: 10,
      label: 'screening rejection threshold tau',
    }),
    degradationLevel: slider(0, 8, 3, {
      step: 0.5,
      label: 'degradation / test sample parameter',
    }),
  })

  const { screeningMetric, threshold, degradationLevel } = state

  const results = useMemo(() => {
    if (screeningMetric === 'laplacian_blur') {
      const sigmas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      const variances = sigmas.map((s) => 650 / Math.pow(1 + 0.65 * s * s, 1.4))
      const cutoff = sigmas.map(() => threshold)

      const currentVariance = 650 / Math.pow(1 + 0.65 * degradationLevel * degradationLevel, 1.4)
      const isRetained = currentVariance >= threshold

      return {
        x: sigmas,
        y: variances,
        baseline: cutoff,
        stat1: `Laplacian Var: ${currentVariance.toFixed(1)}`,
        stat2: `Threshold: ${threshold.toFixed(0)}`,
        stat3: `Screening: ${isRetained ? 'PASS (Retained)' : 'REJECT (Blurry)'}`,
        warning:
          currentVariance < threshold
            ? 'High-frequency gradient energy fell below threshold; edge transitions are degraded by defocus or motion blur'
            : 'Sufficient edge sharpness retained; high-frequency edge energy exceeds screening cut-off',
        axisXLabel: 'Gaussian Blur Kernel Radius (Sigma in Pixels)',
        axisYLabel: 'Discrete Laplacian Variance Var(∇²I)',
        xRange: [0, 8.2] as [number, number],
        yRange: [0, 700] as [number, number],
      }
    } else if (screeningMetric === 'exposure_clipping') {
      const evStops = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
      const clippingPct = evStops.map((ev) => {
        if (ev < 0) {
          return Math.min(85, Math.pow(Math.abs(ev) / 3.0, 2.2) * 85)
        } else {
          return Math.min(90, Math.pow(ev / 3.0, 2.4) * 90)
        }
      })
      const maxClipAllowed = threshold / 10
      const cutoff = evStops.map(() => maxClipAllowed)

      const ev = degradationLevel - 3 // map 0..8 to -3..+5 EV
      const currClip =
        ev < 0
          ? Math.min(85, Math.pow(Math.abs(ev) / 3.0, 2.2) * 85)
          : Math.min(90, Math.pow(Math.min(3, ev) / 3.0, 2.4) * 90)
      const isRetained = currClip <= maxClipAllowed

      return {
        x: evStops,
        y: clippingPct,
        baseline: cutoff,
        stat1: `Exposure Shift: ${ev > 0 ? '+' : ''}${ev.toFixed(1)} EV`,
        stat2: `Clipped Pixels: ${currClip.toFixed(1)}%`,
        stat3: `Screening: ${isRetained ? 'PASS (Acceptable)' : 'REJECT (Severe Clipping)'}`,
        warning:
          currClip > maxClipAllowed
            ? 'Severe dynamic range truncation; significant sensor saturation in darks or highlights destroys semantic contrast'
            : 'Dynamic range within acceptable bounds; pixel intensities distributed across mid-tones without excessive saturation',
        axisXLabel: 'Exposure Compensation Shift (Stops EV)',
        axisYLabel: 'Saturated / Clipped Pixels (%)',
        xRange: [-3.2, 3.2] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (screeningMetric === 'brisque_compression') {
      const qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
      const brisqueScores = qualities.map((q) => 10 + 82 / (1 + Math.exp((q - 40) / 12)))
      const brisqueCutoff = threshold / 6
      const cutoff = qualities.map(() => brisqueCutoff)

      const currQuality = 10 + degradationLevel * 10
      const currBrisque = 10 + 82 / (1 + Math.exp((currQuality - 40) / 12))
      const isRetained = currBrisque <= brisqueCutoff

      return {
        x: qualities,
        y: brisqueScores,
        baseline: cutoff,
        stat1: `JPEG Quality: Q=${currQuality.toFixed(0)}`,
        stat2: `BRISQUE Score: ${currBrisque.toFixed(1)} (0=Pristine, 100=Poor)`,
        stat3: `Screening: ${isRetained ? 'PASS (Clean)' : 'REJECT (Compression Damage)'}`,
        warning:
          currBrisque > brisqueCutoff
            ? 'Mean-subtracted contrast-normalized (MSCN) coefficients deviate strongly from natural scene statistics due to 8x8 DCT ringing and blocking'
            : 'Natural scene statistics preserved; spatial correlation and structural gradients remain intact',
        axisXLabel: 'JPEG Compression Quality Factor (Q)',
        axisYLabel: 'BRISQUE No-Reference Distortion Score (Lower is Better)',
        xRange: [5, 100] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else {
      const sampleIndices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const hammingDistances = [0, 2, 3, 5, 24, 27, 31, 32, 35, 39]
      const hashThreshold = Math.min(30, Math.max(2, Math.round(threshold / 20)))
      const cutoff = sampleIndices.map(() => hashThreshold)

      const currIndex = Math.min(10, Math.max(1, Math.round(degradationLevel) + 1))
      const currDist = hammingDistances[currIndex - 1]
      const isDuplicate = currDist <= hashThreshold

      return {
        x: sampleIndices,
        y: hammingDistances,
        baseline: cutoff,
        stat1: `Pair #${currIndex} Hamming Dist: ${currDist} / 64 bits`,
        stat2: `Dup Threshold: ≤ ${hashThreshold} bits`,
        stat3: `Deduplication: ${isDuplicate ? 'FLAGGED AS DUPLICATE' : 'DISTINCT IMAGE'}`,
        warning: isDuplicate
          ? 'Perceptual hash collision indicates near-identical visual content (rescaling, cropping, or watermark); deduplicate to prevent leakage'
          : 'Low bit correlation indicates orthogonal image contents; distinct instances retained in the dataset',
        axisXLabel: 'Image Pair Candidate (1-4: Near-Duplicates, 5-10: Distinct Pairs)',
        axisYLabel: '64-bit DCT pHash Hamming Distance d_H',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 50] as [number, number],
      }
    }
  }, [screeningMetric, threshold, degradationLevel])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Image Quality Screening: Blur, Exposure, Compression, and Deduplication"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Observed Signal" value={results.stat1} />
          <Readout label="Filter Boundary" value={results.stat2} />
          <Readout label="Pipeline Decision" value={results.stat3} />
          <Readout label="Auditing Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Degradation Metric vs Rejection Boundary">
        <Curve name="Rejection Threshold Boundary" x={results.x} y={results.baseline} slot={2} dashed width={1.8} />
        <Curve
          name={
            screeningMetric === 'laplacian_blur'
              ? 'Laplacian Spatial Variance Var(∇²I)'
              : screeningMetric === 'exposure_clipping'
                ? 'Pixel Saturation Clipping Ratio (%)'
                : screeningMetric === 'brisque_compression'
                  ? 'BRISQUE Distortion Index'
                  : 'Hamming Bit Distance d_H'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Observed Samples" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
