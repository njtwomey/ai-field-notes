import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'

const PEOPLE = 1000
const COLUMNS = 50

/** A population of 1,000 as a grid, split by disease status and test result. Positives are highlighted. */
export function NaturalFrequencies() {
  const [prevalence, setPrevalence] = useState(1)
  const [sensitivity, setSensitivity] = useState(99)
  const [specificity, setSpecificity] = useState(95)

  const result = useMemo(() => {
    const sick = Math.round((PEOPLE * prevalence) / 100)
    const truePos = Math.round((sick * sensitivity) / 100)
    const falseNeg = sick - truePos
    const healthy = PEOPLE - sick
    const falsePos = Math.round((healthy * (100 - specificity)) / 100)
    const trueNeg = healthy - falsePos
    // Lay people out in reading order, grouped so each outcome forms one block.
    const groups: [string, number, Partial<XYSeries>][] = [
      ['true positive', truePos, { slot: 0 }],
      ['missed (false negative)', falseNeg, { slot: 2 }],
      ['false positive', falsePos, { slot: 1 }],
      ['true negative', trueNeg, { muted: true }],
    ]
    let index = 0
    const series = groups.map(([name, count, style]): XYSeries => {
      const cells = Array.from({ length: count }, () => index++)
      return {
        name,
        type: 'scatter',
        x: cells.map((i) => i % COLUMNS),
        y: cells.map((i) => -Math.floor(i / COLUMNS)),
        ...style,
      }
    })
    const exact = (sensitivity * prevalence) / (sensitivity * prevalence + (100 - specificity) * (100 - prevalence))
    return { truePos, falsePos, falseNeg, series, exact }
  }, [prevalence, sensitivity, specificity])

  return (
    <Interactive
      title="1,000 people take the test"
      caption="Each mark is a person. Of everyone who tests positive, what share is sick? Lower the prevalence: the false positives, drawn from the large healthy majority, soon outnumber the true positives."
      controls={
        <>
          <ParamSlider
            label="prevalence"
            value={prevalence}
            onChange={setPrevalence}
            min={0.1}
            max={50}
            step={0.1}
            format={(v) => `${v}%`}
          />
          <ParamSlider
            label="sensitivity P(+ | D)"
            value={sensitivity}
            onChange={setSensitivity}
            min={50}
            max={100}
            step={0.5}
            format={(v) => `${v}%`}
          />
          <ParamSlider
            label="specificity P(− | not D)"
            value={specificity}
            onChange={setSpecificity}
            min={50}
            max={100}
            step={0.5}
            format={(v) => `${v}%`}
          />
        </>
      }
      readout={
        <>
          <Readout label="positives" value={result.truePos + result.falsePos} />
          <Readout label="of which sick" value={result.truePos} />
          <Readout label="P(D | +)" value={`${formatNumber(100 * result.exact)}%`} />
        </>
      }
    >
      <XYChart bare height={240} series={result.series} xRange={[-1, COLUMNS]} yRange={[-PEOPLE / COLUMNS, 1]} />
    </Interactive>
  )
}
