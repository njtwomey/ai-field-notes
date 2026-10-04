import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const PEOPLE = 1000
const COLUMNS = 50

/** A population of 1,000 as a grid, split by disease status and test result. Positives are highlighted. */
export function NaturalFrequencies() {
  const state = useFigureState({
    prevalence: slider(0.1, 50, 1, { step: 0.1, label: 'prevalence', format: (v) => `${v}%` }),
    sensitivity: float(99, { min: 50, max: 100, step: 0.5, label: 'sensitivity P(+ | D)', format: (v) => `${v}%` }),
    specificity: float(95, { min: 50, max: 100, step: 0.5, label: 'specificity P(− | not D)', format: (v) => `${v}%` }),
  })

  const result = useMemo(() => {
    const sick = Math.round((PEOPLE * state.prevalence) / 100)
    const truePos = Math.round((sick * state.sensitivity) / 100)
    const falseNeg = sick - truePos
    const healthy = PEOPLE - sick
    const falsePos = Math.round((healthy * (100 - state.specificity)) / 100)
    const trueNeg = healthy - falsePos
    // Lay people out in reading order, grouped so each outcome forms one block.
    const groups: [string, number, Partial<SeriesSpec>][] = [
      ['true positive', truePos, { slot: 0 }],
      ['missed (false negative)', falseNeg, { slot: 2 }],
      ['false positive', falsePos, { slot: 1 }],
      ['true negative', trueNeg, { muted: true }],
    ]
    let index = 0
    const series = groups.map(([name, count, style]): SeriesSpec => {
      const cells = Array.from({ length: count }, () => index++)
      return {
        name,
        type: 'scatter',
        x: cells.map((i) => i % COLUMNS),
        y: cells.map((i) => -Math.floor(i / COLUMNS)),
        ...style,
      }
    })
    const exact =
      (state.sensitivity * state.prevalence) /
      (state.sensitivity * state.prevalence + (100 - state.specificity) * (100 - state.prevalence))
    return { truePos, falsePos, falseNeg, series, exact }
  }, [state.prevalence, state.sensitivity, state.specificity])

  const xAxis = useAxis({ range: [-1, COLUMNS] })
  const yAxis = useAxis({ range: [-PEOPLE / COLUMNS, 1] })
  return (
    <Figure
      title="1,000 people take the test"
      state={state}
      caption="Each mark is a person. Of everyone who tests positive, what share is sick? Lower the prevalence: the false positives, drawn from the large healthy majority, soon outnumber the true positives."

      readouts={
        <>
          <Readout label="positives" value={result.truePos + result.falsePos} />
          <Readout label="of which sick" value={result.truePos} />
          <Readout label="P(D | +)" value={`${formatNumber(100 * result.exact)}%`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={240} bare>
        {seriesLayers(result.series)}
      </Plot>
    </Figure>
  )
}
