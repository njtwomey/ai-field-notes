import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

const ROUNDS = 30
const ETA = 0.1
const C1 = 0
const C2 = 1
const A1 = 1

/**
 * FedAvg on two equally weighted clients with quadratic losses a_k/2 (w − c_k)². Each round, each client runs s steps of
 * gradient descent from the server model and the server averages the results. With s = 1 this is gradient descent on
 * the global loss; with more local steps the fixed point drifts from the global optimum towards the mean of the
 * clients' own optima.
 */
export function ClientDrift() {
  const steps = useParam(10, { min: 1, max: 50, step: 1 })
  const a2 = useParam(3, { min: 0.5, max: 5, step: 0.1 })

  const { path, fixed } = useMemo(() => {
    const r1 = (1 - ETA * A1) ** steps.value
    const r2 = (1 - ETA * a2.value) ** steps.value
    const local = (w: number, c: number, r: number) => c + r * (w - c)
    const ws = [-1]
    for (let t = 0; t < ROUNDS; t++) {
      const w = ws[ws.length - 1]
      ws.push(0.5 * local(w, C1, r1) + 0.5 * local(w, C2, r2))
    }
    return { path: ws, fixed: ((1 - r1) * C1 + (1 - r2) * C2) / (1 - r1 + (1 - r2)) }
  }, [steps.value, a2.value])
  const optimum = (A1 * C1 + a2.value * C2) / (A1 + a2.value)
  const rounds = path.map((_, i) => i)

  const series: XYSeries[] = [
    { name: 'global optimum', type: 'line', x: [0, ROUNDS], y: [optimum, optimum], dashed: true, emphasis: true },
    { name: 'server model', type: 'line', x: rounds, y: path, slot: 0 },
  ]

  return (
    <Interactive
      title="Client drift in federated averaging"
      caption="Two clients with equal weight hold quadratic losses with optima at 0 and 1 and curvatures 1 and a₂. Each round, both run s local gradient steps with step size 0.1 from the server model, and the server averages. With one local step FedAvg is gradient descent on the global loss and converges to the dashed global optimum. With more local steps it converges faster but to the wrong point, pulled towards 0.5, the plain average of the clients' optima."
      controls={
        <>
          <ParamSlider label="local steps s" param={steps} />
          <ParamSlider label="curvature of client 2, a₂" param={a2} />
        </>
      }
      readout={
        <>
          <Readout label="global optimum" value={formatNumber(optimum)} />
          <Readout label="FedAvg fixed point" value={formatNumber(fixed)} />
          <Readout label="gap" value={formatNumber(fixed - optimum)} />
        </>
      }
    >
      <XYChart
        series={series}
        xRange={[0, ROUNDS]}
        yRange={[-1, 1]}
        xLabel="communication round"
        yLabel="server parameter w"
        ariaLabel="Server model over communication rounds of federated averaging, with the global optimum"
      />
    </Interactive>
  )
}
