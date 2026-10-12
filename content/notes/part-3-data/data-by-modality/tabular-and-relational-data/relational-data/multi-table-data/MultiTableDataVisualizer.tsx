import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MultiTableDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'fanout_row_explosion' | 'deep_feature_synthesis' | 'relational_graph_scaling'>(
      [
        { value: 'fanout_row_explosion', label: 'Denormalization: 1-to-Many Join Fan-Out & Row Explosion (Codd 1970)' },
        {
          value: 'deep_feature_synthesis',
          label: 'Relational Feature Aggregation Depth vs Feature Count (Kanter 2015)',
        },
        { value: 'relational_graph_scaling', label: 'Relational Graph Message Passing: Memory Complexity (Fey 2024)' },
      ],
      'fanout_row_explosion',
      { label: 'relational learning approach' },
    ),
    entityCount: slider(100, 1000, 400, {
      step: 50,
      label: 'root parent entities N_users',
    }),
    fanoutFactor: slider(2, 12, 5, {
      step: 1,
      label: 'child branching factor k',
    }),
  })

  const { visualizerMode, entityCount, fanoutFactor } = state

  const results = useMemo(() => {
    if (visualizerMode === 'fanout_row_explosion') {
      // Depth of relational join path: 1 (Users), 2 (Users -> Orders), 3 (Users -> Orders -> Items), 4 (+ Events)
      const depths = [1, 2, 3, 4]
      // Row count expands as N * k^(depth - 1)
      const explodedRows = depths.map((d) => entityCount * Math.pow(fanoutFactor, d - 1))
      // Normalized root entity count stays constant = N
      const normalizedRows = depths.map(() => entityCount)

      const currExploded = entityCount * Math.pow(fanoutFactor, 2) // depth 3
      const currMultiplier = Math.pow(fanoutFactor, 2)

      return {
        x: depths,
        y: explodedRows,
        baseline: normalizedRows,
        stat1: `Root entities N = ${entityCount}`,
        stat2: `Branching factor k = ${fanoutFactor}`,
        stat3: `Depth-3 rows: ${currExploded.toLocaleString()} (${currMultiplier}x fan-out)`,
        warning:
          fanoutFactor >= 6
            ? 'Severe join fan-out: Naive table flattening multiplies row count by >36x, duplicating parent features'
            : 'Moderate fan-out: Joining 1-to-many child tables alters the fundamental unit of observation',
        axisXLabel: 'Relational Join Path Depth (1=Root, 2=Child, 3=Grandchild, 4=Leaf)',
        axisYLabel: 'Flattened Denormalized Row Count',
        xRange: [0.8, 4.2] as [number, number],
        yRange: [0, Math.max(...explodedRows) * 1.05] as [number, number],
      }
    } else if (visualizerMode === 'deep_feature_synthesis') {
      // Aggregation traversal depths: 1 to 5
      const depths = [1, 2, 3, 4, 5]
      // Base features per table = 6. At depth d, combinations of aggregations (mean, sum, max, count, min, std)
      // DFS creates candidate features exponentially or polynomially with depth
      const features = depths.map((d) => Math.round(6 * d + Math.pow(d, 2.5) * (fanoutFactor * 0.8)))
      const recommendedThreshold = depths.map(() => 150)

      const currFeatures = Math.round(6 * 3 + Math.pow(3, 2.5) * (fanoutFactor * 0.8))

      return {
        x: depths,
        y: features,
        baseline: recommendedThreshold,
        stat1: `Depth 3 generated features: ${currFeatures}`,
        stat2: `Traversal max depth: 5`,
        stat3: currFeatures > 150 ? 'High feature dimensionality' : 'Manageable feature space',
        warning:
          currFeatures > 150
            ? 'Candidate feature explosion: Deep relational traversals generate hundreds of collinear aggregations'
            : 'Controlled aggregation: Deep Feature Synthesis extracts informative cross-table entity summaries',
        axisXLabel: 'Relational Aggregation Traversal Depth (Hops)',
        axisYLabel: 'Engineered Feature Cardinality',
        xRange: [0.8, 5.2] as [number, number],
        yRange: [0, Math.max(...features, 200) * 1.05] as [number, number],
      }
    } else {
      // Relational deep learning memory complexity:
      // In flattened table: row count * column count ~ N * k^2 * 20
      // In relational graph: total nodes and edges ~ N * (1 + k + k^2) * d
      const entityScales = [100, 250, 500, 750, 1000]
      const k = fanoutFactor
      // Normalized graph nodes + edges (in thousands of units)
      const graphFootprint = entityScales.map((n) => Math.round((n * (1 + k + k * k) * 8) / 1000))
      // Flattened dense matrix footprint (in thousands of units)
      const flattenedFootprint = entityScales.map((n) => Math.round((n * Math.pow(k, 2) * 40) / 1000))

      const currGraph = Math.round((entityCount * (1 + k + k * k) * 8) / 1000)
      const currFlat = Math.round((entityCount * Math.pow(k, 2) * 40) / 1000)

      return {
        x: entityScales,
        y: flattenedFootprint,
        baseline: graphFootprint,
        stat1: `Root entities N = ${entityCount}`,
        stat2: `Graph memory: ~${currGraph}k units`,
        stat3: `Flattened memory: ~${currFlat}k units`,
        warning:
          currFlat > currGraph * 2
            ? 'Graph neural networks on relational schemas avoid denormalization memory blowup by operating directly on table edges'
            : 'Direct relational GNN message passing maintains linear memory scaling across table joins',
        axisXLabel: 'Root Entity Population (N_users)',
        axisYLabel: 'Memory Consumption Index (Arbitrary Units x1000)',
        xRange: [80, 1020] as [number, number],
        yRange: [0, Math.max(...flattenedFootprint) * 1.05] as [number, number],
      }
    }
  }, [visualizerMode, entityCount, fanoutFactor])

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
      title="Relational & Multi-Table Data: Join Fan-Out, Feature Synthesis, and Graph Learning Dynamics"
      state={state}
      hoverReadout={false}
      caption="Evaluating learning across relational schemas: Multiplicative fan-out row explosion under naive table joins, aggregation feature explosion in Deep Feature Synthesis, and relational graph message passing."
      readouts={
        <>
          <Readout label="Entity Scope" value={results.stat1} />
          <Readout label="Complexity Scale" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'fanout_row_explosion'
                ? 'Fan-Out Factor'
                : visualizerMode === 'deep_feature_synthesis'
                  ? 'Dimensionality'
                  : 'Flattened Footprint'
            }
            value={results.stat3}
            color={visualizerMode === 'relational_graph_scaling' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Architectural Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Relational Data Representation Profile">
        <Curve
          name={
            visualizerMode === 'fanout_row_explosion'
              ? 'Normalized Root Entity Count (No Duplication)'
              : visualizerMode === 'deep_feature_synthesis'
                ? 'Pruning Threshold Benchmark'
                : 'Relational Graph Message-Passing Footprint'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'fanout_row_explosion'
              ? 'Denormalized Flattened Row Expansion'
              : visualizerMode === 'deep_feature_synthesis'
                ? 'Engineered Feature Cardinality by Depth'
                : 'Flattened Denormalized Matrix Footprint'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Trajectory" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
