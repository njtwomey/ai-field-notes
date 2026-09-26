import type { ReactNode } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

/**
 * Side-by-side facts for case studies: `<SpecTable columns={['GPT-2', 'DeepSeek-V3']} rows={{ Parameters: ['1.5B',
 * '671B'] }} />`.
 */
export function SpecTable({ columns, rows }: { columns: string[]; rows: Record<string, ReactNode[]> }) {
  return (
    <div className="not-prose my-6 overflow-x-auto rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            {columns.map((c) => (
              <TableHead key={c}>{c}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(rows).map(([label, cells]) => (
            <TableRow key={label}>
              <TableCell className="text-muted-foreground">{label}</TableCell>
              {cells.map((cell, i) => (
                <TableCell key={i}>{cell}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
