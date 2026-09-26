import { Heatmap, XYChart, formatNumber } from '@/components/viz'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { Output } from '@/generated/contracts'
import { generatedUrl } from '@/lib/generated'

/**
 * Renders one `emit.*` record. Add a case here whenever a new output kind is added to contracts.py; the switch is
 * exhaustive, so `tsc` fails until it is handled.
 */
export function OutputView({ output, runDir }: { output: Output; runDir: string }) {
  return (
    <figure className="space-y-3">
      {output.title && <figcaption className="text-sm font-medium">{output.title}</figcaption>}
      <OutputBody output={output} runDir={runDir} />
    </figure>
  )
}

function OutputBody({ output, runDir }: { output: Output; runDir: string }) {
  switch (output.kind) {
    case 'text':
      return <pre className="overflow-x-auto rounded-lg border bg-muted/30 p-4 font-mono text-xs">{output.text}</pre>
    case 'metrics':
      return (
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {Object.entries(output.values).map(([k, v]) => (
            <div key={k} className="rounded-lg border px-3 py-2">
              <dt className="text-xs text-muted-foreground">{k}</dt>
              <dd className="font-mono text-lg tabular-nums">{typeof v === 'number' ? formatNumber(v) : v}</dd>
            </div>
          ))}
        </dl>
      )
    case 'table':
      return (
        <div className="max-h-96 overflow-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                {output.columns.map((c) => (
                  <TableHead key={c}>{c}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {output.rows.map((row, i) => (
                <TableRow key={i}>
                  {row.map((cell, j) => (
                    <TableCell key={j} className="font-mono text-xs tabular-nums">
                      {typeof cell === 'number' ? formatNumber(cell) : (cell ?? '')}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )
    case 'chart':
      return (
        <XYChart
          series={output.series.map((s) => ({ ...s, emphasis: s.name === 'centroids' }))}
          xLabel={output.x_label}
          yLabel={output.y_label}
          ariaLabel={output.title ?? undefined}
        />
      )
    case 'heatmap':
      return (
        <Heatmap
          x={output.x}
          y={output.y}
          z={output.z}
          xLabel={output.x_label}
          yLabel={output.y_label}
          overlay={output.overlay}
          ariaLabel={output.title ?? undefined}
        />
      )
    case 'file':
      return output.mime.startsWith('image/') ? (
        <img
          src={generatedUrl(`${runDir}/${output.path}`)}
          alt={output.title ?? output.path}
          className="rounded-lg border"
        />
      ) : (
        <a href={generatedUrl(`${runDir}/${output.path}`)} className="underline">
          {output.path}
        </a>
      )
    default:
      return output satisfies never
  }
}
