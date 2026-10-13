import {
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'

const GAMMA_MIN = 0.05
const GAMMA_MAX = 0.5

/**
 * The witness construction that shows non-decreasing functions from [0, 1] to [0, 1] γ-shatter ⌊1/(2γ)⌋ points:
 * witnesses r_i = (2i − 1)γ, and for each pattern the function sits at r_i + γ or r_i − γ, which never decreases.
 */
export function FatShatteringMonotone() {
  const state = useFigureState({
    gamma: slider(GAMMA_MIN, GAMMA_MAX, 0.1, { step: 0.01, label: 'margin γ' }),
  })
  const gamma = state.gamma
  const d = Math.floor(1 / (2 * gamma) + 1e-9)
  const patterns = 2 ** d
  const [k, setK] = usePlayhead(patterns)
  const pattern = Math.min(k, patterns - 1)
  const bits = Array.from({ length: d }, (_, i) => (pattern >> i) & 1)

  // The construction from the note: point i (1-based) at x = i/(d + 1), witness (2i − 1)γ, value r ± γ by the pattern.
  const xs = bits.map((_, i) => (i + 1) / (d + 1))
  const witnesses = bits.map((_, i) => (2 * i + 1) * gamma)
  const values = bits.map((b, i) => witnesses[i] + (b ? gamma : -gamma))
  const stepX = [0, ...xs.flatMap((x, i) => [x, i + 1 < d ? xs[i + 1] : 1])]
  const stepY = [values[0], ...values.flatMap((v) => [v, v])]
  const bands = xs.map((x, i) => ({ from: [x, witnesses[i] - gamma] as const, to: [x, witnesses[i] + gamma] as const }))
  const ticks = xs.map((x, i) => ({ from: [x - 0.02, witnesses[i]] as const, to: [x + 0.02, witnesses[i]] as const }))
  const above = bits.flatMap((b, i) => (b ? [i] : []))
  const below = bits.flatMap((b, i) => (b ? [] : [i]))

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'f(x)', range: [0, 1] })
  return (
    <Figure
      title="Non-decreasing functions fat-shatter ⌊1/(2γ)⌋ points"
      state={state}
      controls={
        <ControlRow label="pattern">
          <Player
            className="col-span-full"
            value={pattern}
            onChange={setK}
            count={patterns}
            label="pattern b"
            format={(p) => Array.from({ length: d }, (_, i) => (p >> i) & 1).join('')}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="points shattered ⌊1/(2γ)⌋" value={formatNumber(d)} />
          <Readout label="patterns 2ᵈ" value={formatNumber(patterns)} />
          <Readout label="pattern b" value={bits.join('')} />
        </>
      }
      caption="Each grey bar is a witness band from rᵢ − γ to rᵢ + γ around the witness rᵢ = (2i − 1)γ (short tick). For pattern b the non-decreasing step function sits on the top of band i where bᵢ = 1 and on the bottom where bᵢ = 0. Consecutive bands touch, so every pattern is non-decreasing, and the bands fill [0, 1] exactly when d = ⌊1/(2γ)⌋: one more point would need a band beyond 1. Play through the patterns, and drag the top of the first band (the γ handle) to change the margin."
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Segments name="witness bands" segments={bands} width={6} />
        <Segments name="witnesses" segments={ticks} width={2} />
        <Curve name="f" x={stepX} y={stepY} emphasis />
        {above.length > 0 && (
          <Points name="above rᵢ + γ" x={above.map((i) => xs[i])} y={above.map((i) => values[i])} slot={1} size={12} />
        )}
        {below.length > 0 && (
          <Points name="below rᵢ − γ" x={below.map((i) => xs[i])} y={below.map((i) => values[i])} slot={0} size={12} />
        )}
        <Handle
          kind="y"
          at={2 * gamma}
          label="γ"
          onDrag={(y) => state.set('gamma', Math.round(Math.max(GAMMA_MIN, Math.min(GAMMA_MAX, y / 2)) * 100) / 100)}
        />
      </Plot>
    </Figure>
  )
}
