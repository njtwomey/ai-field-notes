import { useState } from 'react'
import { NumberField } from '@lab/controls'
import { Controls } from '@lab/layout'
import { Readout, Readouts } from '@lab/viz'

/** The typed number field in its states: valid, invalid drafts (int, strict and inclusive bounds) and log10. */
export function NumberFieldDemo() {
  const [lr, setLr] = useState(1e-3)
  const [episodes, setEpisodes] = useState(200)
  const [every, setEvery] = useState(10)
  const [gamma, setGamma] = useState(0.95)
  const [tolerance, setTolerance] = useState(1e-6)
  return (
    <div className="flex flex-col gap-4 rounded-xl border bg-card p-4">
      <Controls>
        <NumberField
          label="learning rate · float, gt 0, log10"
          value={lr}
          onChange={setLr}
          type="float"
          gt={0}
          scale="log10"
          suggestions={[1e-4, 3e-4, 1e-3, 3e-3]}
        />
        <NumberField
          label="episodes · int, ge 1, le 10000"
          value={episodes}
          onChange={setEpisodes}
          type="int"
          ge={1}
          le={10_000}
          suggestions={[200, 400, 800, 1600]}
        />
        <NumberField
          label="update every · int, ge 1"
          value={every}
          onChange={setEvery}
          type="int"
          ge={1}
          suggestions={[1, 4, 10, 100]}
        />
        <NumberField
          label="discount γ · ge 0, lt 1, step 0.01"
          value={gamma}
          onChange={setGamma}
          ge={0}
          lt={1}
          step={0.01}
        />
        <NumberField
          label="tolerance · gt 0, log10, step 1"
          value={tolerance}
          onChange={setTolerance}
          gt={0}
          scale="log10"
          step={1}
        />
      </Controls>
      <Controls>
        <NumberField label="invalid · gt 0" value={1e-3} onChange={() => {}} gt={0} scale="log10" initialDraft="0" />
        <NumberField label="invalid · int" value={200} onChange={() => {}} type="int" ge={1} initialDraft="2.5" />
        <NumberField
          label="invalid · le 5000"
          value={200}
          onChange={() => {}}
          type="int"
          ge={1}
          le={5000}
          initialDraft="6000"
        />
        <NumberField label="invalid · not a number" value={0.5} onChange={() => {}} initialDraft="1e" />
        <NumberField label="disabled" value={3e-4} onChange={() => {}} gt={0} scale="log10" disabled />
      </Controls>
      <Readouts>
        <Readout label="learning rate" value={lr} />
        <Readout label="episodes" value={episodes} />
        <Readout label="update every" value={every} />
        <Readout label="γ" value={gamma} />
        <Readout label="tolerance" value={tolerance} />
      </Readouts>
      <p className="text-xs text-muted-foreground">
        Type a value and press Enter: an invalid one stays red with its reason and is not committed; Escape or blur
        reverts. − and + (and ↑/↓) step by the step, or on a log10 scale by half a decade (1e-3 → 3.16e-3 → 1e-2); Shift
        moves ten steps or a whole decade. The buttons clamp to the bounds and stop just inside a strict one. The
        chevron opens the suggested values.
      </p>
    </div>
  )
}
