import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'

const T = 60

export function CvFolds() {
  const initial = useParam(24, { min: 10, max: 40, step: 1 })
  const horizon = useParam(4, { min: 1, max: 10, step: 1 })
  const step = useParam(4, { min: 1, max: 10, step: 1 })
  const gap = useParam(0, { min: 0, max: 5, step: 1 })
  const [windowKind, setWindowKind] = useState<'expanding' | 'sliding'>('expanding')

  const layout = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    const group: number[] = []
    let folds = 0
    let trainPoints = 0
    // Origin n: the model is trained on data up to n and forecasts n + gap + 1 ... n + gap + horizon.
    for (let n = initial.value; n + gap.value + horizon.value <= T; n += step.value) {
      folds++
      const row = folds
      const start = windowKind === 'expanding' ? 1 : n - initial.value + 1
      for (let t = start; t <= n; t++) {
        x.push(t)
        y.push(row)
        group.push(0)
        trainPoints++
      }
      for (let t = n + 1; t <= n + gap.value; t++) {
        x.push(t)
        y.push(row)
        group.push(1)
      }
      for (let t = n + gap.value + 1; t <= n + gap.value + horizon.value; t++) {
        x.push(t)
        y.push(row)
        group.push(2)
      }
    }
    const series: XYSeries[] = [
      { name: 'folds', type: 'scatter', x, y, group, groupNames: ['training', 'gap', 'test'] },
    ]
    return { series, folds, trainPoints }
  }, [initial.value, horizon.value, step.value, gap.value, windowKind])

  return (
    <Interactive
      title="Rolling-origin folds"
      caption={`Each row is one fold over a series of ${T} observations. The model is trained on the training points and forecasts the test points, which always lie after them. An expanding window keeps all past data; a sliding window keeps only the most recent observations, as many as the initial size. A gap between training and test data leaves out observations whose features or targets would overlap the test period. Every fold contributes one forecast at each horizon from 1 to h.`}
      controls={
        <>
          <ParamChoice
            label="training window"
            value={windowKind}
            onChange={setWindowKind}
            options={[
              { value: 'expanding', label: 'expanding' },
              { value: 'sliding', label: 'sliding' },
            ]}
          />
          <ParamSlider label="initial training size" param={initial} format={(v) => String(v)} />
          <ParamSlider label="forecast horizon h" param={horizon} format={(v) => String(v)} withArrows />
          <ParamSlider label="step between origins" param={step} format={(v) => String(v)} withArrows />
          <ParamSlider label="gap" param={gap} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="folds (model fits)" value={String(layout.folds)} />
          <Readout label="test forecasts" value={String(layout.folds * horizon.value)} />
          <Readout label="training points processed" value={String(layout.trainPoints)} />
        </>
      }
    >
      <XYChart series={layout.series} xLabel="time t" yLabel="fold" xRange={[0, T + 1]} height={300} />
    </Interactive>
  )
}
