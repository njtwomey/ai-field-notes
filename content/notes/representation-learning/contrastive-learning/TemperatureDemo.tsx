import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const NEGATIVES = 256
const CURVE_X = Array.from({ length: 201 }, (_, i) => -1 + i / 100)

/** Cosine similarities of 256 negatives to the anchor: mostly unrelated, a few near the positive. */
function draw(seed: number): number[] {
  const r = rng(seed)
  return Array.from({ length: NEGATIVES }, () => Math.max(-1, Math.min(1, 0.1 + 0.2 * r.normal())))
}

/** InfoNCE for one anchor: the positive at similarity sPos against 256 negatives, at temperature τ. */
export function TemperatureDemo() {
  const [seed, setSeed] = useState(1)
  const [logTau, setLogTau] = useState(-1)
  const sPos = useParam(0.6, { min: -1, max: 1, step: 0.01 })
  const tau = 10 ** logTau
  const negs = useMemo(() => draw(seed), [seed])

  const r = useMemo(() => {
    // Softmax over the positive and all negatives, computed relative to the largest logit.
    const top = Math.max(sPos.value, ...negs)
    const ePos = Math.exp((sPos.value - top) / tau)
    const eNeg = negs.map((s) => Math.exp((s - top) / tau))
    const zNeg = eNeg.reduce((a, b) => a + b, 0)
    const pPos = ePos / (ePos + zNeg)
    // Each negative's share of the total gradient weight on negatives (∂L/∂s_j ∝ p_j).
    const share = eNeg.map((e) => e / zNeg)
    const effective = 1 / share.reduce((a, q) => a + q * q, 0)
    const hardest = Math.max(...negs)
    const relative = negs.map((s) => Math.exp((s - hardest) / tau))
    const curve = CURVE_X.map((x) => Math.min(1, Math.exp((x - hardest) / tau)))
    return { loss: -Math.log(pPos), pPos, effective, relative, curve }
  }, [negs, sPos.value, tau])

  const series: XYSeries[] = [
    { name: 'weight ∝ exp(s/τ)', type: 'line', x: CURVE_X, y: r.curve, muted: true },
    { name: 'negatives', type: 'scatter', x: negs, y: r.relative, slot: 0 },
  ]
  const handles: Handle[] = [{ kind: 'x', at: sPos.value, label: 'positive', onDrag: sPos.set }]

  return (
    <Interactive
      title="Temperature decides which negatives matter"
      caption="Each dot is one of 256 negatives, placed at its cosine similarity to the anchor. Its height is its gradient weight in the InfoNCE loss, relative to the hardest negative. At high temperature every negative is pushed away about equally. At low temperature almost all the weight falls on the few most similar negatives. Drag the vertical line to move the positive's similarity."
      controls={
        <>
          <ParamSlider
            label="temperature τ"
            value={logTau}
            onChange={setLogTau}
            min={-2}
            max={0}
            step={0.05}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider label="positive similarity" param={sPos} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New negatives</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="loss (nats)" value={formatNumber(r.loss)} />
          <Readout label="softmax weight on the positive" value={formatNumber(r.pPos)} />
          <Readout label="effective number of negatives" value={formatNumber(r.effective)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="cosine similarity to the anchor"
        yLabel="relative gradient weight"
        xRange={[-1, 1]}
        yRange={[0, 1.05]}
        handles={handles}
        height={300}
      />
    </Interactive>
  )
}
