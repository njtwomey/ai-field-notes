import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'

const XS = linspace(20, 120, 400)

/**
 * Two groups with Gaussian scores. The total variance of a random student's score splits into the average variance
 * within the groups plus the variance of the group means.
 */
export function TotalVariance() {
  const [weightA, setWeightA] = useState(0.3)
  const [meanA, setMeanA] = useState(60)
  const [meanB, setMeanB] = useState(80)
  const [sdA, setSdA] = useState(5)
  const [sdB, setSdB] = useState(4)

  const result = useMemo(() => {
    const weightB = 1 - weightA
    const mean = weightA * meanA + weightB * meanB
    const within = weightA * sdA ** 2 + weightB * sdB ** 2
    const between = weightA * (meanA - mean) ** 2 + weightB * (meanB - mean) ** 2
    const a = XS.map((x) => (weightA * normalPdf((x - meanA) / sdA)) / sdA)
    const b = XS.map((x) => (weightB * normalPdf((x - meanB) / sdB)) / sdB)
    const series: XYSeries[] = [
      { name: 'group A (weighted)', type: 'line', x: XS, y: a, area: true, slot: 0 },
      { name: 'group B (weighted)', type: 'line', x: XS, y: b, area: true, slot: 1 },
      { name: 'all students', type: 'line', x: XS, y: a.map((v, i) => v + b[i]), dashed: true, slot: 2 },
    ]
    return { mean, within, between, series }
  }, [weightA, meanA, meanB, sdA, sdB])

  return (
    <Interactive
      title="Within-group and between-group variance"
      caption="Each group's scores are Gaussian. The dashed curve is the distribution of a randomly chosen student's score, a mixture of the two groups. Its variance is the average within-group variance plus the variance of the group means. Move the means apart and the between-group term grows; shrink the spreads and only the between-group term is left."
      controls={
        <>
          <ParamSlider
            label="share in group A"
            value={weightA}
            onChange={setWeightA}
            min={0.05}
            max={0.95}
            step={0.05}
          />
          <ParamSlider label="group A mean" value={meanA} onChange={setMeanA} min={40} max={100} step={1} />
          <ParamSlider label="group B mean" value={meanB} onChange={setMeanB} min={40} max={100} step={1} />
          <ParamSlider label="group A standard deviation" value={sdA} onChange={setSdA} min={1} max={12} step={0.5} />
          <ParamSlider label="group B standard deviation" value={sdB} onChange={setSdB} min={1} max={12} step={0.5} />
        </>
      }
      readout={
        <>
          <Readout label="E[X]" value={formatNumber(result.mean)} />
          <Readout label="E[var(X | Y)] (within)" value={formatNumber(result.within)} />
          <Readout label="var(E[X | Y]) (between)" value={formatNumber(result.between)} />
          <Readout label="var(X) (total)" value={formatNumber(result.within + result.between)} />
          <Readout
            label="share between groups"
            value={`${Math.round((100 * result.between) / (result.within + result.between))}%`}
          />
        </>
      }
    >
      <XYChart height={300} xLabel="score x" yLabel="density" series={result.series} yRange={[0, undefined]} />
    </Interactive>
  )
}
