import {
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'

/**
 * The N-shattering of k − 1 points by non-decreasing step functions into k labels: candidate labels i and i + 1 at
 * point i, and for each pattern the step function that picks i + bᵢ, which never decreases.
 */
export function NatarajanOrdinal() {
  const state = useFigureState({
    k: int(5, { ge: 3, le: 10, suggestions: [3, 5, 8, 10], label: 'labels k' }),
  })
  const k = state.k
  const d = k - 1
  const patterns = 2 ** d
  const [position, setPosition] = usePlayhead(patterns)
  const pattern = Math.min(position, patterns - 1)
  const bits = Array.from({ length: d }, (_, i) => (pattern >> i) & 1)

  // Point i (1-based) at x = i/(d + 1); f₀ gives it label i, f₁ label i + 1, and the pattern picks one.
  const xs = bits.map((_, i) => (i + 1) / (d + 1))
  const f0 = bits.map((_, i) => i + 1)
  const f1 = bits.map((_, i) => i + 2)
  const chosen = bits.map((b, i) => i + 1 + b)
  const stepX = [0, ...xs.flatMap((x, i) => [x, x, i + 1 < d ? xs[i + 1] : 1])]
  const stepY = [1, ...chosen.flatMap((v, i) => [i === 0 ? 1 : chosen[i - 1], v, v])]

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'label h(x)', range: [0.5, k + 0.5] })
  return (
    <Figure
      title="Ordinal thresholds N-shatter k − 1 points"
      state={state}
      controls={
        <ControlRow label="pattern">
          <Player
            className="col-span-full"
            value={pattern}
            onChange={setPosition}
            count={patterns}
            label="pattern b"
            format={(p) => Array.from({ length: d }, (_, i) => (p >> i) & 1).join('')}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="Natarajan dimension k − 1" value={formatNumber(d)} />
          <Readout label="patterns 2ᵏ⁻¹" value={formatNumber(patterns)} />
          <Readout label="labels chosen" value={chosen.join(' ')} />
        </>
      }
      caption="Point i has two candidate labels, f₀(xᵢ) = i (lower marker) and f₁(xᵢ) = i + 1 (upper marker). For each pattern b the step function takes label i + bᵢ at point i. The chosen labels never decrease, so a non-decreasing step function into {1, …, k} realises every pattern. A k-th point would need a chain of k + 1 distinct labels. Play through the patterns or change k."
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Points name="f₀: label i" x={xs} y={f0} slot={0} size={10} thin />
        <Points name="f₁: label i + 1" x={xs} y={f1} slot={1} size={10} thin />
        <Curve name="h" x={stepX} y={stepY} emphasis />
        <Points name="h(xᵢ)" x={xs} y={chosen} emphasis size={12} />
      </Plot>
    </Figure>
  )
}
