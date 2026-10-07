import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const QS = Array.from({ length: 24 }, (_, i) => i + 1)
/** Fixed cost of one forward call (launch, scheduling, a network hop), seconds. */
const CALL = 0.03
/** Extra cost of the shared path: keeping the prefix state and branching it per question, seconds. */
const SHARE = 0.04
/** Tokens in one question's ending (instructions, criteria, the field selector). */
const SUFFIX = 40

/**
 * A cost model of answering Q questions about one state of S tokens. Reading is parallel over tokens, at R tokens per
 * second; writing costs d seconds per output token, one token after another.
 */
export function DecisionLatency() {
  const state = useFigureState({
    s: int(2000, { min: 50, max: 64000, label: 'state length S (tokens)', suggestions: [240, 2000, 12000] }),
    out: int(20, { min: 1, max: 4000, label: 'text answer length (tokens per question)', suggestions: [5, 20, 500] }),
    rate: float(10000, { min: 500, max: 100000, label: 'read rate R (tokens/s)', suggestions: [2000, 10000, 40000] }),
    decode: float(0.02, { min: 0.002, max: 0.2, label: 'write time d (s per token)', suggestions: [0.01, 0.02, 0.05] }),
    q: slider(1, 24, 8, { step: 1, onChart: true, label: 'questions Q' }),
  })
  const { s, out, rate, decode } = state

  const read = (tokens: number) => CALL + tokens / rate
  const curves = useMemo(() => {
    const generate = QS.map((q) => CALL + (s + q * SUFFIX) / rate + q * out * decode)
    const separate = QS.map((q) => q * (CALL + (s + SUFFIX) / rate))
    const shared = QS.map((q) => CALL + s / rate + SHARE + CALL + (q * SUFFIX) / rate)
    return { generate, separate, shared }
  }, [s, out, rate, decode])

  const q = state.q
  const sep = q * read(s + SUFFIX)
  const sha = read(s) + SHARE + read(q * SUFFIX)
  const gen = read(s + q * SUFFIX) + q * out * decode

  const xAxis = useAxis({ label: 'questions about one state, Q', range: [1, 24], integer: true })
  const yAxis = useAxis({ label: 'latency (s)', log: true, hold: 'union', key: [s, out, rate, decode] })
  return (
    <Figure
      title="Where the time goes in a batch of decisions"
      state={state}
      caption="An illustrative cost model, not a measurement. A text model answers all Q questions in one call and writes its answers token by token. Separate decision reads read the whole state once per question. The shared read reads the state once and then runs every question's short ending against the stored state in one batch, at a fixed bookkeeping cost. Drag Q along the axis. With one question the shared path costs more than it saves, as Nimble's own timing found; the gap to the text model is set by how many tokens it writes."
      readouts={
        <>
          <Readout label="text answers" value={`${formatNumber(gen)} s`} />
          <Readout label="separate reads" value={`${formatNumber(sep)} s`} />
          <Readout label="shared read" value={`${formatNumber(sha)} s`} />
          <Readout label="text ÷ shared" value={`${formatNumber(gen / sha)}×`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve name="text model, writes answers" x={QS} y={curves.generate} slot={1} />
        <Curve name="separate decision reads" x={QS} y={curves.separate} slot={2} />
        <Curve name="shared read, batched endings" x={QS} y={curves.shared} slot={0} />
        <Handle {...state.handle('q', { label: 'Q' })} />
      </Plot>
    </Figure>
  )
}
