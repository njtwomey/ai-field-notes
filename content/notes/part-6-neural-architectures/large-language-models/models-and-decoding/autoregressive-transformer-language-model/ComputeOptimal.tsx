import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

// Hoffmann et al. (2022), approach 3: L(N, D) = E + A / N^α + B / D^β, with training compute C ≈ 6ND.
const E = 1.69
const A = 406.4
const B = 410.7
const ALPHA = 0.34
const BETA = 0.28
const G = ((ALPHA * A) / (BETA * B)) ** (1 / (ALPHA + BETA))

const LOG_N = Array.from({ length: 161 }, (_, i) => 7 + i / 32)
const BUDGETS = [19, 20, 21, 22, 23, 24, 25]

const loss = (n: number, d: number) => E + A / n ** ALPHA + B / d ** BETA
const isoFlop = (logC: number) => LOG_N.map((l) => loss(10 ** l, 10 ** logC / (6 * 10 ** l)))
const optimalN = (logC: number) => G * (10 ** logC / 6) ** (BETA / (ALPHA + BETA))

const sci = (v: number) => {
  const e = Math.floor(Math.log10(v))
  return `${formatNumber(v / 10 ** e)} × 10^${e}`
}

/** Iso-FLOP curves of the Chinchilla loss fit: predicted loss against model size at fixed training compute. */
export function ComputeOptimal() {
  const logC = useParam(23.76, { min: 19, max: 25, step: 0.02 })
  const logN = useParam(11.45, { min: 7, max: 12, step: 0.01 })

  const background = useMemo<XYSeries[]>(
    () => BUDGETS.map((b) => ({ name: 'budgets 10¹⁹ to 10²⁵', type: 'line', x: LOG_N, y: isoFlop(b), muted: true })),
    [],
  )
  const current = useMemo(() => isoFlop(logC.value), [logC.value])

  const nOpt = optimalN(logC.value)
  const dOpt = 10 ** logC.value / (6 * nOpt)
  const n = 10 ** logN.value
  const d = 10 ** logC.value / (6 * n)

  const series: XYSeries[] = [
    ...background,
    { name: 'chosen budget', type: 'line', x: LOG_N, y: current, slot: 0 },
    { name: 'compute-optimal size', type: 'scatter', x: [Math.log10(nOpt)], y: [loss(nOpt, dOpt)], emphasis: true },
    { name: 'chosen size', type: 'scatter', x: [logN.value], y: [loss(n, d)], slot: 1 },
  ]
  const handles: Handle[] = [{ kind: 'x', at: logN.value, label: 'N', onDrag: logN.set }]

  return (
    <Interactive
      title="Compute-optimal model size"
      caption="Each curve holds the training compute C = 6ND fixed and trades parameters N against tokens D, using the loss fit of Hoffmann et al. A small model sees many tokens but cannot use them; a large model sees too few. The black point is the minimum. Drag the vertical line to choose a model size and read off its loss and the tokens it can afford. The default budget is Chinchilla's, about 5.8 × 10²³ FLOPs, and the default size is Gopher's 280 billion parameters."
      controls={
        <>
          <ParamSlider label="log₁₀ compute C (FLOPs)" param={logC} />
          <ParamSlider label="log₁₀ parameters N" param={logN} />
        </>
      }
      readout={
        <>
          <Readout label="optimal N" value={sci(nOpt)} />
          <Readout label="optimal D (tokens)" value={sci(dOpt)} />
          <Readout label="tokens per parameter at optimum" value={formatNumber(dOpt / nOpt)} />
          <Readout
            label="loss at optimum / at chosen N"
            value={`${formatNumber(loss(nOpt, dOpt))} / ${formatNumber(loss(n, d))}`}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="log₁₀ parameters N"
        yLabel="predicted loss (nats per token)"
        xRange={[7, 12]}
        yRange={[1.7, 4]}
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
