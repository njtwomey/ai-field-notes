import { useMemo, useState } from 'react'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { poolAdjacentViolatorsSteps, type PavState } from 'aifn-compute/learning/calibration'
import {
  ControlGroup,
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  choice,
  row,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'

const fmt = (v: number) => formatNumber(v)

type PresetKey = 'worked-example' | 'noisy-sigmoid' | 'plateaus' | 'dips'

type Dataset = {
  label: string
  x: number[]
  y: number[]
}

const PRESETS: Record<PresetKey, Dataset> = {
  'worked-example': {
    label: 'Worked example: (2, 5, 3, 1, 6, 4)',
    x: [0, 1, 2, 3, 4, 5],
    y: [2, 5, 3, 1, 6, 4],
  },
  'noisy-sigmoid': {
    label: 'Noisy sigmoid with local inversions (n = 14)',
    x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    y: [0.8, 1.4, 1.1, 2.8, 2.3, 4.6, 5.2, 4.8, 7.1, 6.9, 8.5, 8.2, 9.4, 9.7],
  },
  plateaus: {
    label: 'Three plateaus with fluctuations (n = 12)',
    x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    y: [2.1, 1.8, 2.4, 1.9, 5.2, 4.7, 5.5, 5.0, 8.3, 7.8, 8.6, 8.4],
  },
  dips: {
    label: 'Concave growth with sharp dip (n = 10)',
    x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    y: [1.5, 3.2, 4.1, 4.8, 2.2, 5.7, 6.4, 5.9, 7.5, 8.0],
  },
}

const PRESET_OPTIONS = (Object.keys(PRESETS) as PresetKey[]).map((k) => ({
  value: k,
  label: PRESETS[k].label,
}))

export function IsotonicPavAnimation() {
  const state = useFigureState({
    data: row('Data', {
      preset: choice(PRESET_OPTIONS, 'worked-example', { label: 'dataset' }),
      increasing: toggle(true, 'monotonically non-decreasing'),
    }),
    view: row('Display', {
      showLinear: toggle(true, 'show unconstrained linear regression (OLS)'),
      showFinal: toggle(true, 'show final isotonic step function'),
    }),
  })

  const presetKey = state.data.preset as PresetKey
  const { increasing } = state.data
  const { showLinear, showFinal } = state.view

  const data = PRESETS[presetKey]
  const n = data.x.length

  // Run the full trace of PAV
  const tr = useMemo(() => {
    return trace(poolAdjacentViolatorsSteps(data.y, { increasing }), undefined, 200, { keep: 'all' })
  }, [data.y, increasing])

  const [stepIndex, setStepIndex] = useState(0)
  const currentStep = Math.min(stepIndex, tr.steps.length - 1)
  const step: PavState = tr.steps[currentStep]

  // Step data unpacking
  const starts = Array.from(toFlat(step.starts))
  const values = Array.from(toFlat(step.values))

  // Ordinary least squares for comparison
  const ols = useMemo(() => {
    const x = data.x
    const y = data.y
    const mx = x.reduce((a, b) => a + b, 0) / n
    const my = y.reduce((a, b) => a + b, 0) / n
    let num = 0
    let den = 0
    for (let i = 0; i < n; i++) {
      num += (x[i] - mx) * (y[i] - my)
      den += (x[i] - mx) ** 2
    }
    const slope = den > 0 ? num / den : 0
    const intercept = my - slope * mx
    const xMin = Math.min(...x)
    const xMax = Math.max(...x)
    return {
      x: [xMin, xMax],
      y: [intercept + slope * xMin, intercept + slope * xMax],
    }
  }, [data, n])

  // Current step segments for the active blocks
  const blockSegments = useMemo(() => {
    const segs: { x: number[]; y: number[]; block: number }[] = []
    const padding = 0.4
    for (let b = 0; b < starts.length; b++) {
      const iStart = starts[b]
      const iEnd = b + 1 < starts.length ? starts[b + 1] - 1 : step.next - 1
      if (iEnd < iStart) continue

      const x0 = data.x[iStart] - (iStart === 0 ? 0 : padding)
      const x1 = data.x[iEnd] + (iEnd === n - 1 ? 0 : padding)
      const yVal = values[b]

      segs.push({
        x: [x0, x1],
        y: [yVal, yVal],
        block: b,
      })
    }
    return segs
  }, [starts, values, step.next, data.x, n])

  // Full final step function curve
  const finalCurve = useMemo(() => {
    const xs: number[] = []
    const ys: number[] = []
    const finalStarts = Array.from(toFlat(tr.final.starts))
    const finalValues = Array.from(toFlat(tr.final.values))

    for (let b = 0; b < finalStarts.length; b++) {
      const iStart = finalStarts[b]
      const iEnd = b + 1 < finalStarts.length ? finalStarts[b + 1] - 1 : n - 1
      const x0 = data.x[iStart] - 0.3
      const x1 = data.x[iEnd] + 0.3
      const y = finalValues[b]

      if (xs.length > 0) {
        // Step vertical connector
        xs.push(x0)
        ys.push(ys[ys.length - 1])
      }
      xs.push(x0, x1)
      ys.push(y, y)
    }
    return { x: xs, y: ys }
  }, [tr.final, data.x, n])

  // Active / processed points vs pending points
  const activePoints = useMemo(() => {
    const processedX: number[] = []
    const processedY: number[] = []
    const pendingX: number[] = []
    const pendingY: number[] = []

    for (let i = 0; i < n; i++) {
      if (i < step.next) {
        processedX.push(data.x[i])
        processedY.push(data.y[i])
      } else {
        pendingX.push(data.x[i])
        pendingY.push(data.y[i])
      }
    }
    return { processedX, processedY, pendingX, pendingY }
  }, [data, step.next, n])

  // Error tracking curve over steps
  const sseCurve = useMemo(() => {
    const xs = tr.steps.map((_, i) => i)
    const ys = tr.steps.map((s) => s.sse)
    return { x: xs, y: ys }
  }, [tr.steps])

  // Axes
  const xRange: [number, number] = [Math.min(...data.x) - 0.5, Math.max(...data.x) + 0.5]
  const yRange: [number, number] = [Math.min(...data.y) - 1, Math.max(...data.y) + 1]

  const xAxis = useAxis({ label: 'covariate x', range: xRange })
  const yAxis = useAxis({ label: 'response y and fit f̂(x)', range: yRange })
  const stepAxis = useAxis({ label: 'PAVA step', range: [0, tr.steps.length - 1] })
  const sseAxis = useAxis({ label: 'SSE Σ w_i(y_i − f̂_i)²', nice: true })

  // Detailed event description
  const eventDescription = useMemo(() => {
    if (step.event === 'start') return 'Initialise: empty stack of blocks.'
    if (step.event === 'done') return `Done! Optimal monotonic projection found in ${tr.steps.length - 1} steps.`
    if (step.event === 'add') {
      const idx = step.next - 1
      return `Add point ${idx} (x = ${data.x[idx]}, y = ${data.y[idx]}) as a new block on top of stack.`
    }
    if (step.event === 'pool') {
      return `Pool violation! Top block violated order; merged adjacent blocks into weighted mean.`
    }
    return ''
  }, [step.event, step.next, data.x, data.y, tr.steps.length])

  return (
    <Figure
      title="Pool Adjacent Violators Algorithm (PAVA) step-by-step"
      purpose="Visualise how isotonic regression solves the order-restricted least squares problem. Observations are pushed onto a stack; whenever adjacent blocks violate monotonicity, they are popped, merged to their weighted mean, and cascaded backwards until order is restored."
      defaultSize="L"
      state={state}
      controls={
        <ControlGroup title="Playback" collapsible={false}>
          <ControlRow label="Algorithm step">
            <Player
              value={currentStep}
              onChange={setStepIndex}
              count={tr.steps.length}
              label="step"
              format={(k) => `step ${k} of ${tr.steps.length - 1}`}
            />
          </ControlRow>
        </ControlGroup>
      }
      readouts={{
        stepInfo: (
          <>
            <Readout label="event" value={step.event.toUpperCase()} />
            <Readout label="active blocks in stack" value={String(starts.length)} />
            <Readout label="points processed" value={`${step.next} of ${n}`} />
            <Readout label="residual sum of squares (SSE)" value={fmt(step.sse)} />
            <Readout label="status" value={step.done ? 'Optimal solution' : eventDescription} />
          </>
        ),
      }}
      caption="Step through the Pool Adjacent Violators Algorithm (PAVA). Circles are training observations (bright: added so far, dimmed: yet to be processed). Solid coloured bars indicate current block means in the stack. Notice how whenever an added point falls below the preceding block (event: POOL), the algorithm merges the two adjacent blocks into their weighted average and cascades backwards if necessary. The lower plot tracks the residual SSE across steps."
    >
      <Plots rows={2} heights={[2.2, 1]}>
        <Plot x={xAxis} y={yAxis}>
          {/* Final isotonic step curve */}
          {showFinal && <Curve name="final isotonic fit" x={finalCurve.x} y={finalCurve.y} slot={0} thin muted />}

          {/* OLS linear regression for comparison */}
          {showLinear && <Curve name="OLS linear regression" x={ols.x} y={ols.y} slot={1} dashed muted />}

          {/* Current active block segments */}
          {blockSegments.map((seg, idx) => (
            <Curve key={`block-${idx}`} name={`block ${idx}`} x={seg.x} y={seg.y} slot={2} width={3} emphasis />
          ))}

          {/* Processed points */}
          <Points
            name="processed observations"
            x={activePoints.processedX}
            y={activePoints.processedY}
            slot={0}
            emphasis
          />

          {/* Unprocessed / pending points */}
          <Points name="pending observations" x={activePoints.pendingX} y={activePoints.pendingY} muted />

          {/* Current point added */}
          {step.event === 'add' && step.next > 0 && (
            <Points name="newly added point" x={[data.x[step.next - 1]]} y={[data.y[step.next - 1]]} slot={3} live />
          )}
        </Plot>

        <Plot x={stepAxis} y={sseAxis} legend={false}>
          <Curve name="SSE" x={sseCurve.x} y={sseCurve.y} slot={0} />
          <Points name="current step SSE" x={[currentStep]} y={[step.sse]} slot={3} live />
        </Plot>
      </Plots>
    </Figure>
  )
}
