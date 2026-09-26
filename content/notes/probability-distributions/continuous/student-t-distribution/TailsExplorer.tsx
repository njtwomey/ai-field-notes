import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, ParamSwitch, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { distribution } from '@/lib/distributions'
import { linspace } from '@/lib/math'

const t = distribution('student-t')
const gaussian = distribution('gaussian')
const STANDARD = { mu: 0, sigma: 1 }

/** Student t with ν degrees of freedom against the standard normal, with an optional log scale to show the tails. */
export function TailsExplorer() {
  const [nu, setNu] = useState(2)
  const [log, setLog] = useState(true)
  const [cut, setCut] = useState(3)

  const series = useMemo((): XYSeries[] => {
    const xs = linspace(-8, 8, 401)
    return [
      { name: `t, ν = ${nu}`, type: 'line', x: xs, y: xs.map((x) => t.density(x, { nu })), slot: 0 },
      { name: 'standard normal', type: 'line', x: xs, y: xs.map((x) => gaussian.density(x, STANDARD)), slot: 1 },
    ]
  }, [nu])

  const tailT = 2 * (1 - t.cdf(cut, { nu }))
  const tailZ = 2 * (1 - gaussian.cdf(cut, STANDARD))

  return (
    <Interactive
      title="Heavy tails"
      caption="On a log scale the Gaussian density is a downward parabola and the t density falls far more slowly, because its tails decay as a power of x. Small ν gives far more mass beyond a few units. As ν grows the two curves merge."
      controls={
        <>
          <ParamSlider label="degrees of freedom ν" value={nu} onChange={setNu} min={0.5} max={50} step={0.5} />
          <ParamSlider label="threshold c" value={cut} onChange={setCut} min={1} max={6} step={0.5} />
          <ParamSwitch label="log density" checked={log} onChange={setLog} />
        </>
      }
      readout={
        <>
          <Readout label="P(|T| > c)" value={formatNumber(tailT)} />
          <Readout label="P(|Z| > c)" value={formatNumber(tailZ)} />
          <Readout label="ratio" value={formatNumber(tailT / tailZ)} />
        </>
      }
    >
      <XYChart
        height={300}
        series={series}
        xLabel="x"
        yLabel={log ? 'density (log scale)' : 'density'}
        yLog={log}
        yRange={log ? [1e-16, 1] : [0, undefined]}
      />
    </Interactive>
  )
}
