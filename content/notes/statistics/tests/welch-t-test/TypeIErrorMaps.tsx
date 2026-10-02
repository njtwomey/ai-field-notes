import { Heatmap, Interactive } from 'aifn-render'
import type { TypeIErrorMaps as Maps } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'

/** Simulated Type I error of the pooled and Welch tests, precomputed by python/mlc/figures/welch.py. */
export function TypeIErrorMaps() {
  const { data, error } = useFigure<Maps>('welch-t-test/type-i-error')
  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data) return null
  // Centre the diverging scale on the nominal α: white is correct, blue too cautious, red too many false positives.
  const range: [number, number] = [0, 2 * data.alpha]
  const map = (grid: Maps['pooled'], title: string) => (
    <div>
      <p className="mb-1 text-center text-sm font-medium">{title}</p>
      <Heatmap
        x={grid.x}
        y={grid.y}
        z={grid.z}
        xLabel={grid.x_label}
        yLabel={grid.y_label}
        valueLabel="Type I error"
        scale="diverging"
        range={range}
        height={300}
      />
    </div>
  )
  return (
    <Interactive
      title="How often each test rejects a true null"
      caption={`Both groups have the same mean, so every rejection is a false positive; a correct test rejects ${
        100 * data.alpha
      }% of the time (the neutral colour). Group 1 has n₁ = ${data.n1}. Across the x-axis, group 2's standard deviation runs from a quarter to four times group 1's. Red means too many false positives, blue too few. Each cell is 10,000 simulated experiments.`}
    >
      <div className="grid gap-4 md:grid-cols-2">
        {map(data.pooled, "Pooled (Student's) t-test")}
        {map(data.welch, "Welch's t-test")}
      </div>
    </Interactive>
  )
}
