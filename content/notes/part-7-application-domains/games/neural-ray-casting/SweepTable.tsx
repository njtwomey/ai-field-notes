import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { WORLDS } from '../_shared/world'
import sweep from './_models/sweep.json'

type Entry = {
  world: string
  model: string
  family: string
  familyLabel: string
  label: string
  params: number
  within10: number
  medianRelative: number
  rmse: number
  trainSeconds: number | null
}

const ENTRIES = sweep as Entry[]
/** Columns grouped by class, in the order the sweep lists them. */
const MODELS = [...new Map(ENTRIES.map((e) => [e.model, e])).values()].sort(
  (a, b) => a.family.localeCompare(b.family) || ENTRIES.indexOf(a) - ENTRIES.indexOf(b),
)
const CLASSES = [...new Map(MODELS.map((m) => [m.family, m.familyLabel])).entries()]
const ROWS = WORLDS.filter((w) => ENTRIES.some((e) => e.world === w.id))

/**
 * The scores of the offline sweep (`python -m mlc.examples.neural_ray_casting sweep`): for each map and trained model,
 * the share of held-out rays within 10 % of the true distance, with the best model of each map in bold.
 */
export function SweepTable() {
  return (
    <figure className="not-prose my-6">
      <Table className="text-xs">
        <TableHeader>
          <TableRow>
            <TableHead rowSpan={2}>map</TableHead>
            {CLASSES.map(([family, label]) => (
              <TableHead
                key={family}
                colSpan={MODELS.filter((m) => m.family === family).length}
                className="border-l text-center"
              >
                {label}
              </TableHead>
            ))}
          </TableRow>
          <TableRow>
            {MODELS.map((m, i) => (
              <TableHead
                key={m.model}
                className={`text-right font-normal ${i === 0 || MODELS[i - 1].family !== m.family ? 'border-l' : ''}`}
              >
                {m.label}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {ROWS.map((w) => {
            const cells = MODELS.map((m) => ENTRIES.find((e) => e.world === w.id && e.model === m.model))
            const best = Math.max(...cells.map((c) => c?.within10 ?? -1))
            return (
              <TableRow key={w.id}>
                <TableCell>{w.label}</TableCell>
                {cells.map((c, i) => (
                  <TableCell
                    key={MODELS[i].model}
                    title={
                      c
                        ? `${c.params.toLocaleString()} parameters · median relative error ${(100 * c.medianRelative).toFixed(1)} %`
                        : undefined
                    }
                    className={`text-right tabular-nums ${i === 0 || MODELS[i - 1].family !== MODELS[i].family ? 'border-l' : ''} ${c && c.within10 === best ? 'font-semibold' : 'text-muted-foreground'}`}
                  >
                    {c ? `${(100 * c.within10).toFixed(1)}` : '–'}
                  </TableCell>
                ))}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      <figcaption className="mt-2 text-xs text-muted-foreground">
        Held-out rays within 10 % of the true distance (%), on 300 views of 48 rays per map, after 3000 training steps.
        Bold: the best model for each map. Hover a score for the model&apos;s size and median relative error.
      </figcaption>
    </figure>
  )
}
