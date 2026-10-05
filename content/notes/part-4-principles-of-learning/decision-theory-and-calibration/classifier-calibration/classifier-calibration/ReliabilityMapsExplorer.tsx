import { useMemo, useState } from 'react'
import { classifierOutputs } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  betaCalibration,
  dirichletCalibration,
  histogramBinning,
  isotonicCalibration,
  plattScaling,
  temperatureScaling,
  topLabelConfidence,
} from 'aifn-compute/learning/calibration'
import { expectedCalibrationError, maximumCalibrationError, reliabilityDiagram } from 'aifn-compute/learning/metrics'
import { softmax } from 'aifn-compute/numerics/special'
import {
  Figure,
  ControlRow,
  NumberSelector,
  Select,
  Plots,
  Plot,
  Curve,
  Histogram,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const flat = (t: Tensor) => Array.from(toFlat(t))
const rowsOf = (t: Tensor, from: number, to: number) => (toRows(t) as number[][]).slice(from, to)

const MAPS = [
  { value: 'temperature', label: 'temperature scaling' },
  { value: 'dirichlet', label: 'Dirichlet calibration' },
  { value: 'histogram', label: 'histogram binning (top label)' },
  { value: 'isotonic', label: 'isotonic (top label)' },
  { value: 'beta', label: 'beta calibration (top label)' },
  { value: 'platt', label: 'Platt scaling (top label)' },
] as const
type MapName = (typeof MAPS)[number]['value']

export function ReliabilityMapsExplorer() {
  const [K, setK] = useState(4)
  const [temperature, setTemperature] = useState(2.5)
  const [map, setMap] = useState<MapName>('temperature')
  const [bins, setBins] = useState(12)

  const n = 2000
  const bias = 0
  const seed = 0

  const r = useMemo(() => {
    const d = classifierOutputs(stream(`calibration/outputs/${seed}`), { n, classes: K, temperature, bias })
    const half = Math.floor(n / 2)
    const y = flat(d.y!)
    const calZ = rowsOf(d.x, 0, half)
    const testZ = rowsOf(d.x, half, n)
    const calY = y.slice(0, half)
    const testY = y.slice(half)
    const calP = softmax(tensor(calZ))
    const testP = softmax(tensor(testZ))
    const before = topLabelConfidence(testP, testY)
    const cal = topLabelConfidence(calP, calY)
    let after: number[]
    const m = map as MapName
    if (m === 'temperature' || m === 'dirichlet') {
      const P =
        m === 'temperature'
          ? temperatureScaling(calZ, calY).apply(testZ)
          : dirichletCalibration(calP, calY).apply(testP)
      after = flat(topLabelConfidence(P, testY).confidence)
    } else {
      const s = flat(cal.confidence)
      const c = flat(cal.correct)
      const t = flat(before.confidence)
      if (m === 'histogram') after = flat(histogramBinning(s, c, { bins }).apply(t))
      else if (m === 'isotonic') after = flat(isotonicCalibration(s, c).apply(t))
      else if (m === 'beta') after = flat(betaCalibration(s, c).apply(t))
      else after = flat(plattScaling(tensor(s), tensor(c)).probability(tensor(t)))
    }
    const correct = flat(before.correct)
    const conf = flat(before.confidence)
    return {
      correct,
      conf,
      after,
      dBefore: reliabilityDiagram(correct, conf, { bins }),
      dAfter: reliabilityDiagram(correct, after, { bins }),
      eceBefore: expectedCalibrationError(correct, conf, { bins }),
      eceAfter: expectedCalibrationError(correct, after, { bins }),
      mceBefore: maximumCalibrationError(correct, conf, { bins }),
      mceAfter: maximumCalibrationError(correct, after, { bins }),
      accuracy: correct.reduce((a, b) => a + b, 0) / correct.length,
    }
  }, [K, temperature, bias, n, seed, map, bins])

  const valid = (d: typeof r.dBefore) => {
    const x = flat(d.x)
    const yv = flat(d.y)
    const keep = x.map((_, i) => Number.isFinite(x[i]) && Number.isFinite(yv[i]))
    return { x: x.filter((_, i) => keep[i]), y: yv.filter((_, i) => keep[i]) }
  }
  const b = valid(r.dBefore)
  const a = valid(r.dAfter)

  const confAxis = useAxis({ label: 'top-label confidence', range: [0, 1] })
  const accAxis = useAxis({ label: 'accuracy in the bin', range: [0, 1] })
  const countAxis = useAxis({ label: 'cases', hold: 'union', key: `${n}/${bins}` })
  const edges = Array.from({ length: bins + 1 }, (_, k) => k / bins)

  return (
    <Figure
      title="Reliability before and after a calibration map"
      purpose="A calibrated classifier is right a fraction p of the time when it says p, so its reliability diagram follows the diagonal; a map fitted on held-out cases moves an overconfident model's points back onto it."
      controls={
        <>
          <ControlRow label="Model & distortion">
            <NumberSelector
              label="Classes K"
              value={K}
              onChange={setK}
              min={2}
              max={10}
              step={1}
              suggestions={[2, 4, 10]}
            />
            <NumberSelector
              label="Overconfidence factor"
              value={temperature}
              onChange={setTemperature}
              min={0.5}
              max={5}
              step={0.5}
              suggestions={[0.5, 1, 2.5, 5]}
            />
          </ControlRow>
          <ControlRow label="Calibration map">
            <Select
              label="Map"
              value={map}
              onChange={(v) => setMap(v as MapName)}
              options={MAPS.map((opt) => ({ value: opt.value, label: opt.label }))}
            />
            <NumberSelector
              label="Bins"
              value={bins}
              onChange={setBins}
              min={5}
              max={25}
              step={1}
              suggestions={[5, 10, 12, 20]}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'test performance': (
          <>
            <Readout label="Accuracy" value={fmt(r.accuracy)} />
            <Readout label="ECE before → after" value={`${fmt(r.eceBefore)} → ${fmt(r.eceAfter)}`} />
            <Readout label="MCE before → after" value={`${fmt(r.mceBefore)} → ${fmt(r.mceAfter)}`} />
          </>
        ),
      }}
      caption="The model's logits are the true logits times the overconfidence factor, so its confidence runs ahead of its accuracy and the grey points fall below the diagonal. Every map is fitted on the calibration half and judged on the test half. Temperature scaling undoes a pure scale error; histogram binning and isotonic are free-form steps; beta and Platt fit smooth sigmoid curves."
    >
      <Plots rows={2} heights={[72, 28]} hoverGroup>
        <Plot x={confAxis} y={accAxis}>
          <Curve name="perfect calibration" x={[0, 1]} y={[0, 1]} dashed emphasis />
          <Curve name="before" x={b.x} y={b.y} muted showPoints />
          <Curve name="after" x={a.x} y={a.y} slot={0} showPoints width={2.5} />
        </Plot>
        <Plot x={confAxis} y={countAxis}>
          <Histogram name="confidence before" values={r.conf} bins={edges} range={[0, 1]} normalize="count" muted />
          <Histogram name="confidence after" values={r.after} bins={edges} range={[0, 1]} normalize="count" slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
