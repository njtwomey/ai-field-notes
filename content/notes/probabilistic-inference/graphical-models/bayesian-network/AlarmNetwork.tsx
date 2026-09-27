import { useState } from 'react'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import { Interactive, ParamSwitch, Readout } from '@/components/viz'

// The worked example's numbers: p(B = 1), p(E = 1) and p(A = 1 | B, E).
const P_B = 0.001
const P_E = 0.002
const P_A = (b: number, e: number) =>
  [
    [0.001, 0.29],
    [0.94, 0.95],
  ][b][e]

/** p(B = 1 | evidence), by summing the joint p(B) p(E) p(A | B, E) over the unobserved variables. */
function posteriorBurglary(alarm: boolean, quake: boolean): number {
  const mass = [0, 0]
  for (const b of [0, 1])
    for (const e of quake ? [1] : [0, 1]) {
      const joint = (b ? P_B : 1 - P_B) * (e ? P_E : 1 - P_E) * (alarm ? P_A(b, e) : 1)
      mass[b] += joint
    }
  return mass[1] / (mass[0] + mass[1])
}

/** The burglary–earthquake–alarm collider, with the posterior of a burglary under each choice of evidence. */
export function AlarmNetwork() {
  const [alarm, setAlarm] = useState(true)
  const [quake, setQuake] = useState(false)
  return (
    <Interactive
      title="Explaining away in the alarm network"
      caption="Burglary B and earthquake E are independent causes of the alarm A. A shaded node is observed. Observing A raises the probability of a burglary; observing E as well explains the alarm away. Observing E without A leaves B at its prior."
      controls={
        <>
          <ParamSwitch label="observe A = 1" checked={alarm} onChange={setAlarm} />
          <ParamSwitch label="observe E = 1" checked={quake} onChange={setQuake} />
        </>
      }
      readout={<Readout label="p(B = 1 | evidence)" value={posteriorBurglary(alarm, quake).toPrecision(3)} />}
    >
      <Diagram
        spec={{
          unit: 56,
          nodes: [
            variable('B', 0, 0, '$B$', { highlight: true }),
            variable('E', 2.4, 0, '$E$', { filled: quake }),
            variable('A', 1.2, 1.6, '$A$', { filled: alarm }),
          ],
          edges: [link('B', 'A'), link('E', 'A')],
        }}
        ariaLabel="Burglary and earthquake both point into alarm"
      />
    </Interactive>
  )
}
