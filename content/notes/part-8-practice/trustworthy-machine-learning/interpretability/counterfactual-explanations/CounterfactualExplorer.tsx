import { useMemo, useState } from 'react'
import { feasibilityTask } from 'aifn-methods/data/synthetic'
import { comparisonModel, fullBatchComparison } from 'aifn-methods/neural/full-batch'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import {
  denseFunction,
  denseOutput,
  diverseCounterfactuals,
  faceGraph,
  faceSearch,
  fromMlpParams,
  growingSpheres,
  medianAbsoluteDeviation,
  wachterCounterfactual,
  type Actionability,
  type DenseNetwork,
} from 'aifn/learning/explain'
import { sigmoid } from 'aifn/numerics/special'
import { multivariateKde } from 'aifn/probability/stats'
import {
  ControlGroup,
  Contours,
  Curve,
  Figure,
  Handle,
  NumberSelector,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  Select,
  Segments,
  Switch,
  useAxis,
} from 'aifn-render'

const XR: [number, number] = [-3, 3]
const YR: [number, number] = [-2.2, 2.4]
const GX = Array.from({ length: 61 }, (_, i) => XR[0] + ((XR[1] - XR[0]) * i) / 60)
const GY = Array.from({ length: 47 }, (_, i) => YR[0] + ((YR[1] - YR[0]) * i) / 46)
const GRID = fromData(Float64Array.from(GY.flatMap((y) => GX.map((x) => [x, y])).flat()), [GX.length * GY.length, 2])

const sigmoidOf = (z: Float64Array) => Float64Array.from(z, (v) => sigmoid(v) as number)

const fmt = (v: number | null | undefined, digits = 3) =>
  v !== null && v !== undefined && Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '—'

const GRAPH_OPTIONS = [
  { value: 'kde', label: 'ε-graph with KDE density weights' },
  { value: 'knn', label: 'k-NN graph with local density weights' },
  { value: 'epsilon', label: 'ε-graph with Euclidean lengths' },
]

export function CounterfactualExplorer() {
  // ── 1. Data and Model Controls ──
  const [pointCount, setPointCount] = useState(300)
  const [dataSeed, setDataSeed] = useState(0)
  const [hiddenWidth, setHiddenWidth] = useState(16)
  const [trainSteps, setTrainSteps] = useState(300)

  // ── 2. FACE Search Controls ──
  const [graphType, setGraphType] = useState<'kde' | 'knn' | 'epsilon'>('kde')
  const [epsilon, setEpsilon] = useState(0.45)
  const [densityThreshold, setDensityThreshold] = useState(0.25)
  const [targetProbability, setTargetProbability] = useState(0.75)

  // ── 3. Actionability Constraints ──
  const [fixX1, setFixX1] = useState(false)
  const [fixX2, setFixX2] = useState(false)
  const [upX2, setUpX2] = useState(false)

  // ── 4. Method Overlays ──
  const [showWachter, setShowWachter] = useState(true)
  const [showDice, setShowDice] = useState(true)
  const [showFace, setShowFace] = useState(true)
  const [showEdges, setShowEdges] = useState(false)

  // ── 5. Query Point Selection ──
  const [query, setQuery] = useState<[number, number]>([-1.6, 0.6])

  // ── 6. Growing Spheres Controls ──
  const [gsRadius, setGsRadius] = useState(0.4)
  const [gsSamples, setGsSamples] = useState(250)
  const [gsStep, setGsStep] = useState(0)

  // ── Data Generation ──
  const data = useMemo(() => feasibilityTask(stream(`cf-${dataSeed}`), { n: pointCount }), [dataSeed, pointCount])
  const X = useMemo(() => Float64Array.from(toFlat(data.x)), [data])
  const y = useMemo(() => Array.from(toFlat(data.y!)), [data])
  const n = y.length

  // ── Model Training ──
  const { net } = useMemo(() => {
    const { unravel } = comparisonModel({
      inputs: 2,
      width: hiddenWidth,
      depth: 2,
      activation: 'tanh',
    })
    let lastSnap = null
    for (const snap of fullBatchComparison(data, {
      task: 'classification',
      network: { width: hiddenWidth, depth: 2, activation: 'tanh' },
      optimisers: ['adam'],
      iterations: trainSteps,
      adamStep: 0.02,
      batchSize: 64,
      l2: 1e-3,
      seed: dataSeed,
      checkpoints: 20,
    })) {
      lastSnap = snap
    }
    if (!lastSnap || lastSnap.runs.length === 0) return { net: null, accuracy: 0 }
    const run = lastSnap.runs[0]
    const lastCp = run.checkpoints[run.checkpoints.length - 1]
    const trainedNet: DenseNetwork = fromMlpParams(unravel(lastCp.theta) as object[], 'tanh')
    return { net: trainedNet, accuracy: run.score }
  }, [data, hiddenWidth, trainSteps, dataSeed])

  // Scale (MAD) and Actionability constraints
  const scale = useMemo(() => medianAbsoluteDeviation(data.x), [data])
  const constraints = useMemo((): Actionability => {
    const immutable = [fixX1 ? 0 : -1, fixX2 ? 1 : -1].filter((i) => i >= 0)
    return { immutable, increasing: upX2 ? [1] : [] }
  }, [fixX1, fixX2, upX2])

  // ── Decision Field & Point Probabilities ──
  const field = useMemo(() => {
    if (!net) return null
    const p = sigmoidOf(denseOutput(net, GRID))
    return GY.map((_, r) => Array.from(p.subarray(r * GX.length, (r + 1) * GX.length)))
  }, [net])

  const probability = useMemo(() => (net ? sigmoidOf(denseOutput(net, data.x)) : null), [net, data])

  // ── Relative KDE Density Field ──
  const kde = useMemo(() => {
    const top = Math.max(...Array.from(toFlat(multivariateKde(data.x, data.x).logDensity)))
    const at = (P: Tensor) =>
      Float64Array.from(toFlat(multivariateKde(data.x, P).logDensity), (l) => Math.min(1, Math.exp(l - top)))
    const g = at(GRID)
    return { at, grid: GY.map((_, r) => Array.from(g.subarray(r * GX.length, (r + 1) * GX.length))) }
  }, [data])

  // Probability at query
  const queryProb = useMemo(() => {
    if (!net) return null
    const P = fromData(Float64Array.from(query), [1, 2])
    return sigmoidOf(denseOutput(net, P))[0]
  }, [net, query])

  // ── FACE Graph & Dijkstra Search ──
  const graph = useMemo(
    () =>
      faceGraph(data.x, query, {
        graph: graphType,
        epsilon,
        k: 8,
        constraints,
      }),
    [data, query, graphType, epsilon, constraints],
  )

  const faceResult = useMemo(
    () =>
      graph && probability
        ? faceSearch(graph, probability, {
            predictionThreshold: targetProbability,
            densityThreshold,
          })
        : null,
    [graph, probability, targetProbability, densityThreshold],
  )

  // ── Wachter & DiCE Recourse ──
  const gradientMethods = useMemo(() => {
    if (!net) return null
    const logit = denseFunction(net)
    const tp = targetProbability
    const logitTarget = Math.log(tp / (1 - tp))
    const wachter = wachterCounterfactual(logit, query, {
      target: logitTarget,
      tolerance: 0.2,
      scale,
      constraints,
      steps: 80,
      rate: 0.05,
    })
    const dice = diverseCounterfactuals(
      logit,
      query,
      stream(`dice-${Math.round(query[0] * 100)}-${Math.round(query[1] * 100)}`),
      {
        count: 3,
        steps: 250,
        scale,
        constraints,
      },
    )
    return { wachter, dice }
  }, [net, query, targetProbability, scale, constraints])

  // Point projection helper
  const pts = (rows: number[]) => rows.map((i) => (i === n ? query : [X[2 * i], X[2 * i + 1]]))
  const facePath = faceResult && faceResult.index >= 0 ? pts(faceResult.path) : null
  const wachterPath = gradientMethods
    ? Array.from({ length: gradientMethods.wachter.path.shape[0] }, (_, t) =>
        toFlat(gradientMethods.wachter.path).slice(2 * t, 2 * t + 2),
      )
    : null
  const diceRows = gradientMethods
    ? Array.from({ length: 3 }, (_, k) =>
        Array.from(toFlat(gradientMethods.dice.counterfactuals).slice(2 * k, 2 * k + 2)),
      )
    : []

  // ── Counterfactual Summary Metrics ──
  const ends = useMemo(() => {
    if (!net || !probability) return []
    const list: { name: string; at: number[] }[] = []
    if (gradientMethods) list.push({ name: 'Wachter', at: Array.from(gradientMethods.wachter.counterfactual) })
    if (gradientMethods) diceRows.forEach((r, k) => list.push({ name: `DiCE ${k + 1}`, at: r }))
    if (faceResult?.counterfactual) list.push({ name: 'FACE', at: Array.from(faceResult.counterfactual) })
    if (list.length === 0) return []

    const P = fromData(Float64Array.from(list.flatMap((e) => e.at)), [list.length, 2])
    const dens = kde.at(P)
    const prob = sigmoidOf(denseOutput(net, P))
    return list.map((e, k) => ({
      ...e,
      density: dens[k],
      probability: prob[k],
      distance: Math.hypot(e.at[0] - query[0], e.at[1] - query[1]),
    }))
  }, [gradientMethods, diceRows, faceResult, net, probability, kde, query])

  // ── FACE Edge Segments ──
  const edgeSegments = useMemo(() => {
    if (!graph || !showEdges) return []
    const P = (i: number) => (i === n ? query : [X[2 * i], X[2 * i + 1]])
    return graph.edges
      .filter(([a, b]) => a < b || graph.edges.length < 3000)
      .map(([a, b]) => ({ from: P(a) as [number, number], to: P(b) as [number, number] }))
  }, [graph, showEdges, X, n, query])

  // ── Growing Spheres Simulation ──
  const gsResult = useMemo(() => {
    if (!net) return null
    const predict = (Z: Tensor) => Float64Array.from(denseOutput(net, Z), (v) => (v > 0 ? 1 : 0))
    return growingSpheres(predict, query, stream(`gs-${Math.round(query[0] * 100)}-${Math.round(query[1] * 100)}`), {
      radius: gsRadius,
      samples: gsSamples,
      constraints,
    })
  }, [net, query, gsRadius, gsSamples, constraints])

  const gsLayers = gsResult?.layers ?? []
  const activeGsStep = Math.min(gsStep, Math.max(0, gsLayers.length - 1))
  const lastGsLayer = gsLayers[activeGsStep]
  const isFinalGsStep = activeGsStep === gsLayers.length - 1
  const gsSampleCoords = gsResult ? toFlat(gsResult.samples) : []

  const circle = (rad: number) => {
    const t = Array.from({ length: 121 }, (_, k) => (2 * Math.PI * k) / 120)
    return {
      x: t.map((a) => query[0] + rad * Math.cos(a)),
      y: t.map((a) => query[1] + rad * Math.sin(a)),
    }
  }

  // Snap to nearest dataset point when clicked
  const handlePlotClick = (clicked: [number, number]) => {
    let best = 0
    let minDist = Infinity
    for (let i = 0; i < n; i++) {
      const d2 = (X[2 * i] - clicked[0]) ** 2 + (X[2 * i + 1] - clicked[1]) ** 2
      if (d2 < minDist) {
        minDist = d2
        best = i
      }
    }
    setQuery([X[2 * best], X[2 * best + 1]])
  }

  const ax = useAxis({ label: 'x₁', range: XR, nice: false })
  const ay = useAxis({ label: 'x₂', range: YR, nice: false, equal: ax })

  const gsAx = useAxis({ label: 'x₁', range: XR, nice: false })
  const gsAy = useAxis({ label: 'x₂', range: YR, nice: false, equal: gsAx })

  return (
    <div className="space-y-8">
      {/* ── Figure 1: Gradient Recourse vs Manifold Paths (FACE) ── */}
      <Figure
        title="Counterfactual Recourse: Shortest Change vs Feasible Manifold Path"
        purpose="Wachter et al. and DiCE find the nearest points classified as the target class, which can lie in empty, out-of-distribution space; FACE walks only along short edges between observed data points, weighted to prefer dense regions, ensuring a realistic, reachable counterfactual."
        defaultSize="XL"
        controls={
          <div className="flex flex-col gap-3">
            <ControlGroup title="1 · Data and Architecture">
              <NumberSelector
                label="Sample Count n"
                value={pointCount}
                onChange={setPointCount}
                min={100}
                max={1000}
                step={50}
                suggestions={[200, 300, 600]}
              />
              <NumberSelector
                label="Random Seed"
                value={dataSeed}
                onChange={setDataSeed}
                min={0}
                max={50}
                step={1}
                suggestions={[0, 1, 2, 5]}
              />
              <NumberSelector
                label="Hidden Units"
                value={hiddenWidth}
                onChange={setHiddenWidth}
                min={4}
                max={64}
                step={4}
                suggestions={[8, 16, 32]}
              />
              <NumberSelector
                label="Training Steps"
                value={trainSteps}
                onChange={setTrainSteps}
                min={100}
                max={1500}
                step={50}
                suggestions={[200, 300, 600]}
              />
            </ControlGroup>

            <ControlGroup title="2 · FACE Feasibility Graph">
              <Select
                label="Graph Type"
                value={graphType}
                onChange={(v) => setGraphType(v as 'kde' | 'knn' | 'epsilon')}
                options={GRAPH_OPTIONS}
              />
              <NumberSelector
                label="Max Edge Length ε"
                value={epsilon}
                onChange={setEpsilon}
                min={0.1}
                max={1.5}
                step={0.05}
                suggestions={[0.3, 0.45, 0.7]}
              />
              <NumberSelector
                label="Min Density Threshold t_d"
                value={densityThreshold}
                onChange={setDensityThreshold}
                min={0.01}
                max={0.9}
                step={0.05}
                suggestions={[0.1, 0.25, 0.5]}
              />
              <NumberSelector
                label="Target Probability t_p"
                value={targetProbability}
                onChange={setTargetProbability}
                min={0.5}
                max={0.99}
                step={0.05}
                suggestions={[0.65, 0.75, 0.9]}
              />
            </ControlGroup>

            <ControlGroup title="3 · Actionability Constraints">
              <Switch label="x₁ Immutable" checked={fixX1} onChange={setFixX1} />
              <Switch label="x₂ Immutable" checked={fixX2} onChange={setFixX2} />
              <Switch label="x₂ May Only Rise (Recourse)" checked={upX2} onChange={setUpX2} />
            </ControlGroup>

            <ControlGroup title="4 · Visible Methods & Overlays">
              <Switch label="Wachter (Gradient L1)" checked={showWachter} onChange={setShowWachter} />
              <Switch label="DiCE (Diverse DPP)" checked={showDice} onChange={setShowDice} />
              <Switch label="FACE (Feasible Graph)" checked={showFace} onChange={setShowFace} />
              <Switch label="FACE Graph Edges" checked={showEdges} onChange={setShowEdges} />
            </ControlGroup>
          </div>
        }
        readouts={{
          'Query Point x (drag handle or click data point)': (
            <>
              <Readout label="Coordinates (x₁, x₂)" value={`(${fmt(query[0], 3)}, ${fmt(query[1], 3)})`} />
              <Readout label="P(y = 1) at x" value={queryProb !== null ? fmt(queryProb, 3) : '—'} />
              <Readout
                label="FACE Reachable Candidates"
                value={faceResult ? String(faceResult.candidates.length) : '—'}
              />
              <Readout
                label="FACE Path Cost"
                value={faceResult && faceResult.index >= 0 ? fmt(faceResult.cost, 3) : '—'}
              />
            </>
          ),
          'Methods Comparison: P(y = 1) · Distance · Relative Density': (
            <>
              {ends.map((e) => (
                <Readout
                  key={e.name}
                  label={e.name}
                  value={`${fmt(e.probability, 2)} · ${fmt(e.distance, 2)} · ${fmt(e.density, 2)}`}
                />
              ))}
              {faceResult && faceResult.index < 0 && (
                <Readout label="FACE" value="No reachable path meeting density & probability" />
              )}
            </>
          ),
        }}
        caption="Feasibility dataset: class 0 (left blob) and class 1 (right blob) joined by a high-density data corridor along the lower arc; the central gap is completely empty (off-manifold). Background: fitted MLP probability P(y = 1) with decision boundary at 0.5. Thin contour: relative KDE density threshold t_d. Drag the circular query handle or click any point to set x. Wachter minimises prediction error plus L1 distance, cutting directly across the empty gap. DiCE discovers 3 diverse alternatives, also in the empty space. FACE (Poyiadzi et al., 2020) navigates exclusively through observed data along high-density graph paths via Dijkstra's algorithm. Turn on 'x₂ may only rise' to forbid moving through the downward corridor."
      >
        <Plot x={ax} y={ay} onPlotClick={handlePlotClick}>
          {field && <Raster x={GX} y={GY} z={field} range={[0, 1]} valueLabel="P(y = 1)" fillOpacity={0.35} boundary />}
          <Contours x={GX} y={GY} z={kde.grid} levels={[Math.max(0.01, densityThreshold)]} labels={false} muted />
          {edgeSegments.length > 0 && <Segments segments={edgeSegments} muted width={0.6} />}
          <Points
            name="Data"
            x={Array.from({ length: n }, (_, i) => X[2 * i])}
            y={Array.from({ length: n }, (_, i) => X[2 * i + 1])}
            group={y}
            groupNames={['Class 0', 'Class 1']}
            thin
          />
          {showWachter && wachterPath && (
            <Curve
              name="Wachter Trajectory"
              x={wachterPath.map((p) => p[0])}
              y={wachterPath.map((p) => p[1])}
              slot={3}
              width={2}
            />
          )}
          {showWachter && gradientMethods && (
            <Points
              name="Wachter CF"
              x={[gradientMethods.wachter.counterfactual[0]]}
              y={[gradientMethods.wachter.counterfactual[1]]}
              slot={3}
              size={11}
            />
          )}
          {showDice && gradientMethods && (
            <Segments
              name="DiCE Recourse"
              segments={diceRows.map((r) => ({ from: query, to: r as [number, number] }))}
              slot={4}
              dashed
              width={1.5}
            />
          )}
          {showDice && gradientMethods && (
            <Points
              name="DiCE CFs"
              x={diceRows.map((r) => r[0])}
              y={diceRows.map((r) => r[1])}
              slot={4}
              size={10}
              shape={2}
            />
          )}
          {showFace && facePath && (
            <Curve
              name="FACE Path"
              x={facePath.map((p) => p[0])}
              y={facePath.map((p) => p[1])}
              slot={5}
              width={2.5}
              showPoints
            />
          )}
          <Handle kind="point" at={query} onDrag={setQuery} label="x" />
        </Plot>
      </Figure>

      {/* ── Figure 2: Growing Spheres Concentric Shell Search ── */}
      <Figure
        title="Growing Spheres: Shell Generation and Feature Sparsification"
        purpose="Growing Spheres (Laugel et al., 2018) samples uniformly inside concentric spherical shells around x until an enemy instance (opposite class) is discovered; it then sparsifies the change by resetting features back to x whenever the prediction remains inverted."
        defaultSize="L"
        controls={
          <div className="flex flex-col gap-3">
            <ControlGroup title="1 · Spherical Search">
              <NumberSelector
                label="Initial Radius η"
                value={gsRadius}
                onChange={setGsRadius}
                min={0.1}
                max={3.0}
                step={0.05}
                suggestions={[0.2, 0.4, 0.8]}
              />
              <NumberSelector
                label="Samples Per Shell m"
                value={gsSamples}
                onChange={setGsSamples}
                min={50}
                max={2000}
                step={50}
                suggestions={[100, 250, 500]}
              />
            </ControlGroup>
            <ControlGroup title="2 · Shell Playback">
              <Player
                label="Concentric Shell"
                value={activeGsStep}
                onChange={setGsStep}
                count={Math.max(1, gsLayers.length)}
                format={(p) => `Shell ${p + 1} of ${gsLayers.length}`}
              />
            </ControlGroup>
          </div>
        }
        readouts={{
          'Search Shell Progress': (
            <>
              <Readout label="Total Shells Evaluated" value={String(gsLayers.length)} />
              <Readout label="Model Forward Calls" value={gsResult ? String(gsResult.evaluations) : '—'} />
              <Readout
                label="Current Shell Radius"
                value={lastGsLayer ? `${fmt(lastGsLayer.inner, 2)} < r ≤ ${fmt(lastGsLayer.outer, 2)}` : '—'}
              />
              <Readout label="Enemies Discovered in Shell" value={lastGsLayer ? String(lastGsLayer.enemies) : '—'} />
            </>
          ),
          'Discovered Recourse': (
            <>
              <Readout
                label="Nearest Enemy"
                value={gsResult?.enemy ? `(${fmt(gsResult.enemy[0], 3)}, ${fmt(gsResult.enemy[1], 3)})` : '—'}
              />
              <Readout
                label="Sparse Enemy"
                value={gsResult?.sparse ? `(${fmt(gsResult.sparse[0], 3)}, ${fmt(gsResult.sparse[1], 3)})` : '—'}
              />
            </>
          ),
        }}
        caption="Growing Spheres evaluates concentric shells of thickness η growing outward from x. When enemies are found inside the initial radius, it halves η; otherwise it increments shell radiuses until opposite-class instances are found. At the final shell, the nearest enemy is projected onto coordinate axes to test whether individual feature changes can be dropped (sparse enemy)."
      >
        <Plot x={gsAx} y={gsAy}>
          <Points
            name="Observed Data"
            x={Array.from({ length: X.length / 2 }, (_, i) => X[2 * i])}
            y={Array.from({ length: X.length / 2 }, (_, i) => X[2 * i + 1])}
            muted
            thin
          />
          {gsLayers.slice(0, activeGsStep + 1).map((l, k) => {
            const c = circle(l.outer)
            return <Curve key={k} name="Shell Boundary" x={c.x} y={c.y} muted thin silent />
          })}
          {isFinalGsStep && gsResult && (
            <Points
              name="Final Shell Samples"
              x={Array.from({ length: gsSampleCoords.length / 2 }, (_, i) => gsSampleCoords[2 * i])}
              y={Array.from({ length: gsSampleCoords.length / 2 }, (_, i) => gsSampleCoords[2 * i + 1])}
              group={net ? Array.from(denseOutput(net, gsResult.samples), (v) => (v > 0 ? 1 : 0)) : null}
              groupNames={['Class 0', 'Class 1']}
              thin
            />
          )}
          <Points name="Query x" x={[query[0]]} y={[query[1]]} emphasis size={12} />
          {isFinalGsStep && gsResult?.enemy && (
            <Points name="Nearest Enemy" x={[gsResult.enemy[0]]} y={[gsResult.enemy[1]]} emphasis shape={3} size={12} />
          )}
          {isFinalGsStep && gsResult?.sparse && (
            <Points
              name="Sparse Enemy"
              x={[gsResult.sparse[0]]}
              y={[gsResult.sparse[1]]}
              emphasis
              shape={2}
              size={12}
            />
          )}
        </Plot>
      </Figure>
    </div>
  )
}
