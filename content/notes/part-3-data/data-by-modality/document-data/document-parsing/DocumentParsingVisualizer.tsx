import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DocumentParsingVisualizer() {
  const state = useFigureState({
    parsingMode: choice<'xycut_reading_order' | 'table_grid' | 'rag_noise_impact'>(
      [
        {
          value: 'xycut_reading_order',
          label: 'Reading Order: Multi-Column XY-Cut Projection Profile (Ha 1995)',
        },
        {
          value: 'table_grid',
          label: 'Table Structure: Coordinate Jitter vs Cell Bounding Box Alignment F1',
        },
        {
          value: 'rag_noise_impact',
          label: 'Downstream Impact: Extraction Noise vs Dense Retrieval Cosine Sim (Blecher 2023)',
        },
      ],
      'xycut_reading_order',
      { label: 'document parsing challenge' },
    ),
    param1: slider(5, 50, 20, {
      step: 5,
      label: 'column gutter width / coordinate jitter / noise injection rate',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'cutting threshold / table row count / embedding penalty',
    }),
  })

  const { parsingMode, param1, param2 } = state

  const results = useMemo(() => {
    if (parsingMode === 'xycut_reading_order') {
      // Horizontal page X-coordinate from 0 to 600 points (standard letter/A4 width)
      const xPositions = [0, 50, 100, 150, 200, 250, 275, 290, 300, 310, 325, 350, 400, 450, 500, 550, 600]
      // Projection profile density: Left column (50..280), Gutter (280..320), Right column (320..550)
      const gutterCenter = 300
      const gutterHalfWidth = param1 // 5 to 50 px

      const projectionDensity = xPositions.map((x) => {
        if (x < 50 || x > 550) return 0
        const distToCenter = Math.abs(x - gutterCenter)
        if (distToCenter < gutterHalfWidth) {
          // Gutter valley
          return Math.max(0, 5 - (gutterHalfWidth - distToCenter) * 0.4)
        }
        // Dense text column body
        return 75 + Math.sin(x * 0.08) * 12
      })

      const thresholdCut = param2 * 5 // 5 to 50
      const thresholdCurve = xPositions.map(() => thresholdCut)

      const valleyDepth = Math.min(...projectionDensity.slice(6, 11))
      const canSplit = valleyDepth <= thresholdCut
      const gutterWidthActual = (gutterHalfWidth * 2).toFixed(0)

      return {
        x: xPositions,
        y: projectionDensity,
        baseline: thresholdCurve,
        stat1: `Gutter Gap: ${gutterWidthActual} pt`,
        stat2: `Valley Min: ${valleyDepth.toFixed(1)} glyphs`,
        stat3: `Column Partition: ${canSplit ? 'VALID SPLIT (Column-Major)' : 'FAILED (Row-Major Interleaving)'}`,
        warning: canSplit
          ? 'Clear spatial projection valley detected; recursive vertical cut segments page into isolated columns, preserving natural reading order'
          : 'Narrow gutter or spanning banner blocks vertical cut; naive stream extraction interleaves left/right column lines into incoherent text',
        axisXLabel: 'Document Page Horizontal Coordinate X (PostScript Points pt)',
        axisYLabel: 'Horizontal Projection Histogram Energy (Glyph Overlap Count)',
        xRange: [-10, 610] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (parsingMode === 'table_grid') {
      // Coordinate bounding box jitter sigma from 0 to 20 px
      const jitters = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
      // Table Structure Extraction F1 score (%)
      const structureF1 = jitters.map((j) => {
        const drop = Math.pow(j / 10, 1.7) * 28 + param2 * 1.5
        return Math.max(20, Math.min(98, 98 - drop))
      })
      // Cell Content Text Extraction Accuracy (%)
      const textAcc = jitters.map((j) => {
        const drop = Math.pow(j / 10, 1.4) * 20
        return Math.max(35, 96 - drop)
      })

      const currJitter = (param1 / 50) * 20
      const currDrop = Math.pow(currJitter / 10, 1.7) * 28 + param2 * 1.5
      const currF1 = Math.max(20, Math.min(98, 98 - currDrop))
      const currText = Math.max(35, 96 - Math.pow(currJitter / 10, 1.4) * 20)

      return {
        x: jitters,
        y: structureF1,
        baseline: textAcc,
        stat1: `Bounding Box Jitter: ±${currJitter.toFixed(1)} pt`,
        stat2: `Table Structure F1: ${currF1.toFixed(1)}%`,
        stat3: `Cell Text Integrity: ${currText.toFixed(1)}%`,
        warning:
          currJitter > 8
            ? 'Border jitter merges adjacent table columns or splits merged cells; tabular attributes become hopelessly cross-contaminated'
            : 'Sub-pixel cell alignment correctly reconstructs rectangular row/column adjacency graphs and spans',
        axisXLabel: 'Cell Boundary Coordinate Jitter Sigma (Points pt)',
        axisYLabel: 'Tabular Reconstruction Performance Metric (%)',
        xRange: [0, 21] as [number, number],
        yRange: [15, 105] as [number, number],
      }
    } else {
      // Noise injection rate (%) from 0% to 50%
      const noiseRates = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
      // Dense embedding cosine similarity with ground-truth semantic passage
      const cosineSim = noiseRates.map((n) => Math.max(0.35, 0.94 - Math.pow(n / 50, 1.3) * 0.52))
      // Language model downstream task accuracy / BLEU score (%)
      const downstreamAcc = noiseRates.map((n) => Math.max(15, 88 - Math.pow(n / 50, 1.6) * 65))

      const currNoise = param1 // 5% to 50%
      const currSim = Math.max(0.35, 0.94 - Math.pow(currNoise / 50, 1.3) * 0.52)
      const currAcc = Math.max(15, 88 - Math.pow(currNoise / 50, 1.6) * 65)

      return {
        x: noiseRates,
        y: downstreamAcc,
        baseline: cosineSim.map((s) => s * 100), // scaled to percentage
        stat1: `Parsing Noise Rate: ${currNoise.toFixed(0)}%`,
        stat2: `Embedding Cosine Sim: ${currSim.toFixed(3)}`,
        stat3: `Downstream QA Acc: ${currAcc.toFixed(1)}%`,
        warning:
          currNoise > 20
            ? 'Severe extraction errors (un-dehyphenated words, repeated running headers, scrambled reading order) degrade semantic embedding alignment'
            : 'Clean document structure preserves sentence-level co-reference and dense vector retrieval accuracy',
        axisXLabel: 'Document Parsing Artifact Injection Rate (%)',
        axisYLabel: 'Downstream Retrieval Sim (x100) / QA Model Accuracy (%)',
        xRange: [-2, 52] as [number, number],
        yRange: [10, 100] as [number, number],
      }
    }
  }, [parsingMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Document Parsing: Multi-Column XY-Cut, Table Grids, and Extraction Noise"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Layout Parameter" value={results.stat1} />
          <Readout label="Structural Precision" value={results.stat2} />
          <Readout label="Extraction Decision" value={results.stat3} />
          <Readout label="Parsing Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Document Layout Parsing and Downstream Degradation">
        <Curve
          name={
            parsingMode === 'xycut_reading_order'
              ? 'Projection Cut Threshold Boundary'
              : parsingMode === 'table_grid'
                ? 'Cell Text Ingestion Accuracy (%)'
                : 'Embedding Vector Cosine Similarity (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            parsingMode === 'xycut_reading_order'
              ? 'Horizontal Spatial Projection Profile Energy'
              : parsingMode === 'table_grid'
                ? 'Table Structure Reconstruction F1 (%)'
                : 'Downstream Question-Answering Accuracy (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Layout Data Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
