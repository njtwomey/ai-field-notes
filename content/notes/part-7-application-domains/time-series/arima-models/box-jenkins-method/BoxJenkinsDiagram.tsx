import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'data', x: 0, y: 1, w: 1.8, h: 0.8, label: 'series', tone: 'neutral' },
    { id: 'id', x: 3.6, y: 1, w: 3.2, h: 1.3, label: 'identify\ntransform, difference,\nread ACF and PACF', tone: 0 },
    { id: 'est', x: 8.2, y: 1, w: 3, h: 1.3, label: 'estimate\nmaximum likelihood,\ncompare by AICc', tone: 0 },
    { id: 'chk', x: 13, y: 1, w: 3, h: 1.3, label: 'check residuals\nACF, Ljung–Box,\nnormality', tone: 0 },
    { id: 'dec', x: 13, y: 3.5, shape: 'pill', w: 2.8, h: 0.8, label: 'white noise?', tone: 'neutral' },
    { id: 'fc', x: 16.6, y: 3.5, w: 1.8, h: 0.8, label: 'forecast', tone: 1 },
  ],
  edges: [
    { from: 'data', to: 'id' },
    { from: 'id', to: 'est', label: 'candidates' },
    { from: 'est', to: 'chk' },
    { from: 'chk', to: 'dec' },
    { from: 'dec', to: 'fc', label: 'yes' },
    { from: 'dec:w', to: 'id:s', via: [[3.6, 3.5]], dashed: true, label: 'no: revise the model' },
  ],
}

/** The Box–Jenkins cycle: identify, estimate, check, and go round again until the residuals look like noise. */
export function BoxJenkinsDiagram() {
  return (
    <Interactive
      title="The Box–Jenkins cycle"
      caption="Identification proposes a few candidate orders from the plots and correlograms of the transformed and differenced series. Estimation fits each and ranks them. Diagnostic checking asks whether the chosen model's residuals are white noise. If they are not, the pattern left in them suggests how to revise the model, and the cycle repeats."
    >
      <Diagram
        spec={spec}
        ariaLabel="Box-Jenkins cycle: series to identification, estimation and residual checking; if the residuals are white noise, forecast; otherwise return to identification"
      />
    </Interactive>
  )
}
