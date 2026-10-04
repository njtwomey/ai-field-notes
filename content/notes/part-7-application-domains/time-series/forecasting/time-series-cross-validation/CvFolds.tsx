import { useMemo } from 'react'
import { choice, Figure, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'

const T = 60

export function CvFolds() {
  const state = useFigureState({
    windowKind: choice<'expanding' | 'sliding'>(
      [
        { value: 'expanding', label: 'expanding' },
        { value: 'sliding', label: 'sliding' },
      ],
      'expanding',
      { label: 'training window' },
    ),
    initial: int(24, { min: 10, max: 40, step: 1, label: 'initial training size', format: (v) => String(v) }),
    horizon: int(4, { min: 1, max: 10, step: 1, label: 'forecast horizon h', format: (v) => String(v) }),
    step: int(4, { min: 1, max: 10, step: 1, label: 'step between origins', format: (v) => String(v) }),
    gap: int(0, { min: 0, max: 5, step: 1, label: 'gap', format: (v) => String(v) }),
  })

  const layout = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    const group: number[] = []
    let folds = 0
    let trainPoints = 0
    // Origin n: the model is trained on data up to n and forecasts n + gap + 1 ... n + gap + horizon.
    for (let n = state.initial; n + state.gap + state.horizon <= T; n += state.step) {
      folds++
      const row = folds
      const start = state.windowKind === 'expanding' ? 1 : n - state.initial + 1
      for (let t = start; t <= n; t++) {
        x.push(t)
        y.push(row)
        group.push(0)
        trainPoints++
      }
      for (let t = n + 1; t <= n + state.gap; t++) {
        x.push(t)
        y.push(row)
        group.push(1)
      }
      for (let t = n + state.gap + 1; t <= n + state.gap + state.horizon; t++) {
        x.push(t)
        y.push(row)
        group.push(2)
      }
    }
    const series = [{ name: 'folds', x, y, group, groupNames: ['training', 'gap', 'test'] }] as const
    return { series, folds, trainPoints }
  }, [state.initial, state.horizon, state.step, state.gap, state.windowKind])

  const xAxis = useAxis({ label: 'time t', range: [0, T + 1] })
  const yAxis = useAxis({ label: 'fold', hold: 'union' })
  return (
    <Figure
      title="Rolling-origin folds"
      state={state}
      caption={`Each row is one fold over a series of ${T} observations. The model is trained on the training points and forecasts the test points, which always lie after them. An expanding window keeps all past data; a sliding window keeps only the most recent observations, as many as the initial size. A gap between training and test data leaves out observations whose features or targets would overlap the test period. Every fold contributes one forecast at each horizon from 1 to h.`}

      readouts={
        <>
          <Readout label="folds (model fits)" value={String(layout.folds)} />
          <Readout label="test forecasts" value={String(layout.folds * state.horizon)} />
          <Readout label="training points processed" value={String(layout.trainPoints)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Points {...layout.series[0]} />
      </Plot>
    </Figure>
  )
}
