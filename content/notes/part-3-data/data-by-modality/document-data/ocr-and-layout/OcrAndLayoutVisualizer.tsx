import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function OcrAndLayoutVisualizer() {
  const state = useFigureState({
    ocrMode: choice<'cer_wer_cascade' | 'layout_embeddings' | 'key_value_binding'>(
      [
        {
          value: 'cer_wer_cascade',
          label: 'Error Cascade: Character Error (CER) vs Word Error (WER) & Subword Explosion',
        },
        {
          value: 'layout_embeddings',
          label: '2D Layout Positional Encodings: Spatial Normalization [0, 1000] (Xu 2020)',
        },
        {
          value: 'key_value_binding',
          label: 'Form Key-Value Association: Spatial Proximity vs LayoutLMv3 (Jaume 2019)',
        },
      ],
      'cer_wer_cascade',
      { label: 'ocr layout dimension' },
    ),
    param1: slider(1, 20, 5, {
      step: 1,
      label: 'CER error rate (%) / field width / spatial distance',
    }),
    param2: slider(3, 10, 5, {
      step: 1,
      label: 'average word length / box height / attention strength',
    }),
  })

  const { ocrMode, param1, param2 } = state

  const results = useMemo(() => {
    if (ocrMode === 'cer_wer_cascade') {
      // CER rates from 0% to 20%
      const cerRates = [0, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20]
      const wordLen = param2 // average characters per word L (e.g. 5)
      // WER = 1 - (1 - CER)^L
      const werRates = cerRates.map((c) => (1 - Math.pow(1 - c / 100, wordLen)) * 100)
      // Subword token fragmentation count per word: 1.0 (clean) up to 2.8 (broken into fragments)
      const subwordFragments = cerRates.map((c) => 1.0 + (c / 100) * (wordLen - 1) * 1.5)

      const currCer = param1
      const currWer = (1 - Math.pow(1 - currCer / 100, wordLen)) * 100
      const currFrags = 1.0 + (currCer / 100) * (wordLen - 1) * 1.5

      return {
        x: cerRates,
        y: werRates,
        baseline: subwordFragments.map((f) => f * 20), // scaled to %
        stat1: `CER: ${currCer.toFixed(1)}% (L=${wordLen} chars)`,
        stat2: `Downstream WER: ${currWer.toFixed(1)}%`,
        stat3: `Subword Explosion: ${currFrags.toFixed(2)}x tokens/word`,
        warning:
          currWer > 25
            ? 'Even a modest character error rate triggers a catastrophic cascade in word errors, fragmenting standard terms into out-of-vocabulary subwords'
            : 'Low character corruption preserves word-level token boundaries and dictionary representations',
        axisXLabel: 'OCR Character Error Rate CER (%)',
        axisYLabel: 'Resulting Word Error Rate WER (%) / Token Expansion (x20)',
        xRange: [0, 21] as [number, number],
        yRange: [0, 80] as [number, number],
      }
    } else if (ocrMode === 'layout_embeddings') {
      // Normalized document page coordinate X: 0 to 1000
      const coordinates = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
      // 2D Spatial Positional Embedding similarity with field anchor at x=200
      const anchorX = 200
      const spatialWidth = param1 * 25 // 25 to 500
      // Gaussian spatial proximity embedding response
      const spatialResponse = coordinates.map((x) => 100 * Math.exp(-Math.pow((x - anchorX) / (spatialWidth + 50), 2)))
      // 1D sequential reading order distance baseline
      const readingOrderDistance = coordinates.map((x) => Math.max(0, 100 - Math.abs(x - anchorX) / 10))

      const currBoxW = param1 * 25
      const currBoxH = param2 * 20

      return {
        x: coordinates,
        y: spatialResponse,
        baseline: readingOrderDistance,
        stat1: `Bounding Box: [${anchorX}, 150, ${anchorX + currBoxW}, ${150 + currBoxH}]`,
        stat2: `Normalized Span: ${currBoxW} x ${currBoxH} pt`,
        stat3: `Spatial Embeddings: 2D Box Aware`,
        warning:
          'LayoutLM projects [x0, y0, x1, y1] onto 1000-bin coordinate lookup tables, providing 2D geometric inductive biases orthogonal to 1D token position',
        axisXLabel: 'Normalized Document Page Coordinate Axis (0 to 1000)',
        axisYLabel: '2D Spatial Attention Affinity Score (%)',
        xRange: [-20, 1020] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Key-value spatial distance in pixels: 10 to 200 pt
      const distances = [10, 25, 50, 75, 100, 125, 150, 175, 200]
      // Pure geometric heuristic (nearest neighbor) link accuracy: drops sharply with distance
      const geometricAcc = distances.map((d) => Math.max(10, 96 - Math.pow(d / 80, 1.8) * 45))
      // Multimodal LayoutLMv3 attention linking: robust across whitespace variations
      const multimodalAcc = distances.map((d) => {
        const drop = Math.pow(d / 150, 1.3) * 18
        return Math.max(65, Math.min(98, 98 - drop + param2 * 0.5))
      })

      const currDist = param1 * 10
      const currGeom = Math.max(10, 96 - Math.pow(currDist / 80, 1.8) * 45)
      const currMulti = Math.max(65, Math.min(98, 98 - Math.pow(currDist / 150, 1.3) * 18 + param2 * 0.5))
      const margin = currMulti - currGeom

      return {
        x: distances,
        y: multimodalAcc,
        baseline: geometricAcc,
        stat1: `Key-Value Gap: ${currDist} pt`,
        stat2: `LayoutLMv3 F1: ${currMulti.toFixed(1)}%`,
        stat3: `Naive Geometric F1: ${currGeom.toFixed(1)}% (+${margin.toFixed(1)}% advantage)`,
        warning:
          currDist > 100
            ? 'Large whitespace gaps cause naive geometric heuristics to erroneously link keys to neighboring columns; multimodal transformers resolve visual headers reliably'
            : 'Compact form layouts permit reliable key-value pairing under both geometric and transformer decoders',
        axisXLabel: 'Physical Distance between Key and Value Fields (Points pt)',
        axisYLabel: 'Key-Value Entity Linking F1 Score (%)',
        xRange: [5, 205] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [ocrMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="OCR and Layout: Recognition Errors, 2D Positional Embeddings, and Form Linking"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Input Configuration" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Downstream Impact" value={results.stat3} />
          <Readout label="Architectural Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="OCR Noise Cascade and Spatial Entity Alignment">
        <Curve
          name={
            ocrMode === 'cer_wer_cascade'
              ? 'Subword Token Expansion (x20)'
              : ocrMode === 'layout_embeddings'
                ? '1D Sequential Reading Order Baseline'
                : 'Naive Geometric Proximity F1 (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            ocrMode === 'cer_wer_cascade'
              ? 'Cascaded Word Error Rate WER (%)'
              : ocrMode === 'layout_embeddings'
                ? '2D Spatial Proximity Attention Affinity (%)'
                : 'LayoutLMv3 Multimodal Relation Linking F1 (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Observed Configurations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
