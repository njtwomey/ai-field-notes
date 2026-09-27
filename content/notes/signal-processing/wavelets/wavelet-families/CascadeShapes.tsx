import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, useParam } from '@/components/viz'
import { FILTERS, VANISHING_MOMENTS, cascade, type Family } from '../_shared/wavelets'

/**
 * Scaling function φ and wavelet ψ of the Haar and Daubechies families, computed by the cascade algorithm: iterating
 * the two-scale equation from a unit impulse.
 */
export function CascadeShapes() {
  const [family, setFamily] = useState<Family>('db2')
  const iterations = useParam(7, { min: 1, max: 9, step: 1 })

  const r = useMemo(() => cascade(FILTERS[family], iterations.value), [family, iterations.value])
  const L = FILTERS[family].length

  return (
    <Interactive
      title="Daubechies scaling functions and wavelets"
      caption="The scaling function φ and wavelet ψ, drawn by the cascade algorithm: start from a single impulse and repeatedly upsample and filter with √2·h, which iterates the two-scale equation. After a few iterations the samples settle on the limit functions, supported on [0, L − 1] for a filter of length L. Haar gives boxes. Daubechies wavelets with more vanishing moments are longer and smoother, but none of them is symmetric. The db2 functions are continuous but have no continuous derivative: their Hölder exponent is about 0.55."
      controls={
        <>
          <ParamChoice
            label="family"
            value={family}
            onChange={setFamily}
            options={[
              { value: 'haar', label: 'Haar (db1)' },
              { value: 'db2', label: 'db2' },
              { value: 'db3', label: 'db3' },
              { value: 'db4', label: 'db4' },
            ]}
          />
          <ParamSlider label="cascade iterations" param={iterations} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="filter length L" value={L} />
          <Readout label="support" value={`[0, ${L - 1}]`} />
          <Readout label="vanishing moments" value={VANISHING_MOMENTS[family]} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={[{ name: 'φ (scaling function)', type: 'line', x: r.t, y: r.phi, slot: 0 }]}
          xLabel="t"
          yLabel="φ(t)"
          height={260}
        />
        <XYChart
          series={[{ name: 'ψ (wavelet)', type: 'line', x: r.t, y: r.psi, slot: 1 }]}
          xLabel="t"
          yLabel="ψ(t)"
          height={260}
        />
      </div>
    </Interactive>
  )
}
