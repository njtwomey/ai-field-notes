import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DigitalImagesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'gamma_transfer' | 'jpeg_compression' | 'bitdepth_levels'>(
      [
        {
          value: 'gamma_transfer',
          label: 'Color Management: Physical Linear Flux vs sRGB Gamma Transfer (Gonzalez 2018)',
        },
        {
          value: 'jpeg_compression',
          label: 'Lossy Compression: JPEG Quality Q vs Compression Ratio & Artifacts (Wallace 1991)',
        },
        { value: 'bitdepth_levels', label: 'Quantization: Bit Depth B vs Gradient False Contouring Levels' },
      ],
      'gamma_transfer',
      { label: 'digital image property' },
    ),
    gammaValue: slider(1.0, 2.6, 2.2, {
      step: 0.2,
      label: 'gamma exponent value gamma',
    }),
    jpegQuality: slider(10, 95, 75, {
      step: 5,
      label: 'JPEG compression quality factor Q',
    }),
  })

  const { visualizerMode, gammaValue, jpegQuality } = state

  const results = useMemo(() => {
    if (visualizerMode === 'gamma_transfer') {
      // Linear luminance input from 0.0 to 1.0 (11 sample points)
      const inputLuminance = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      // Gamma encoded pixel value: V = L^(1 / gamma)
      const encodedValues = inputLuminance.map((L) => Math.pow(L, 1 / gammaValue) * 255)
      // Linear identity reference: V = L * 255
      const linearReference = inputLuminance.map((L) => L * 255)

      // At L = 0.214, sRGB value is 0.5 * 255 = 128
      const midGreyPhysical = Math.pow(0.5, gammaValue) * 100

      return {
        x: inputLuminance.map((l) => l * 100),
        y: encodedValues,
        baseline: linearReference,
        stat1: `Gamma exponent: ${gammaValue.toFixed(1)}`,
        stat2: `Mid-grey (128) physical flux: ${midGreyPhysical.toFixed(1)}%`,
        stat3: 'Perceptual Uniformity',
        warning:
          gammaValue > 1.0
            ? 'sRGB allocates over 50% of 8-bit codes to the darkest 21.4% of physical photons to match human eye sensitivity'
            : 'Linear luminance: Physically accurate photon flux, but suffers severe dark-region quantization banding in 8-bit',
        axisXLabel: 'Physical Linear Scene Luminance (% Photon Flux)',
        axisYLabel: 'Digital Stored Pixel Value (0 to 255 uint8)',
        xRange: [-2, 102] as [number, number],
        yRange: [0, 265] as [number, number],
      }
    } else if (visualizerMode === 'jpeg_compression') {
      // Quality factors Q from 10 to 95
      const qValues = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]
      // Compression ratio: e.g. at Q=10, 45:1; at Q=75, 12:1; at Q=95, 4:1
      const compressionRatios = qValues.map((q) => 55 - q * 0.52)
      // High-frequency blocking / artifact index (0 to 100)
      const artifactIndex = qValues.map((q) => Math.max(2, Math.round(100 * Math.exp(-0.045 * q))))

      const currRatio = 55 - jpegQuality * 0.52
      const currArtifacts = Math.max(2, Math.round(100 * Math.exp(-0.045 * jpegQuality)))

      return {
        x: qValues,
        y: compressionRatios,
        baseline: artifactIndex,
        stat1: `Quality Q = ${jpegQuality}`,
        stat2: `Compression ratio: ~${currRatio.toFixed(0)}:1`,
        stat3: `Artifact severity: ${currArtifacts}%`,
        warning:
          jpegQuality < 50
            ? 'Severe DCT blocking: 8x8 block boundary discontinuities disrupt fine convolutional edge filters'
            : jpegQuality >= 80
              ? 'High-quality JPEG: Artifacts imperceptible; preserves fine texture details for classification'
              : 'Standard web JPEG: Good balance between file size and perceptual quality',
        axisXLabel: 'JPEG Quality Factor Setting (Q)',
        axisYLabel: 'Compression Ratio (:1) / Artifact Index (%)',
        xRange: [5, 100] as [number, number],
        yRange: [0, 60] as [number, number],
      }
    } else {
      // Bit depths B from 2 to 16
      const bits = [2, 3, 4, 6, 8, 10, 12, 14, 16]
      // Quantization levels: 2^B
      // Dynamic range: 20 * log10(2^B) = 6.02 * B
      const dynRange = bits.map((b) => b * 6.02)
      const target8Bit = bits.map(() => 48.16)

      return {
        x: bits,
        y: dynRange,
        baseline: target8Bit,
        stat1: 'Quantization depths: 2 to 16 bits',
        stat2: '8-bit standard: 256 levels (48 dB)',
        stat3: '16-bit medical: 65,536 levels (96 dB)',
        warning:
          'Medical imaging (DICOM / CT / MRI) requires 12 to 16 bits; clamping to 8-bit destroys low-contrast soft tissue margins',
        axisXLabel: 'Channel Bit Depth B (Bits / Color Channel)',
        axisYLabel: 'Dynamic Range (dB)',
        xRange: [1.5, 16.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, gammaValue, jpegQuality])

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
      title="Digital Images: sRGB Gamma Transfer, JPEG Compression Dynamics, and Bit Depth Scaling"
      state={state}
      hoverReadout={false}
      caption="Foundations of digital image representations: Physical linear luminance vs sRGB gamma encoding, JPEG rate-distortion and 8x8 blocking artifacts, and dynamic range scaling across bit depths."
      readouts={
        <>
          <Readout label="Representation Metric" value={results.stat1} />
          <Readout label="Dynamic Property" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'gamma_transfer'
                ? 'Encoding Mode'
                : visualizerMode === 'jpeg_compression'
                  ? 'Artifact Index'
                  : 'Medical Standard'
            }
            value={results.stat3}
            color={visualizerMode === 'jpeg_compression' ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
          <Readout label="CV Practice Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Digital Image Encoding Profile">
        <Curve
          name={
            visualizerMode === 'gamma_transfer'
              ? 'Linear Identity Reference (1:1 Photon Flux)'
              : visualizerMode === 'jpeg_compression'
                ? '8x8 DCT Block Artifact Severity Index'
                : '8-Bit Standard Reference Dynamic Range (48 dB)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'gamma_transfer'
              ? 'Gamma-Compressed Stored Value V = L^(1/gamma)'
              : visualizerMode === 'jpeg_compression'
                ? 'File Compression Ratio (:1)'
                : 'Dynamic Range (6.02 * B dB)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Measurement" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
