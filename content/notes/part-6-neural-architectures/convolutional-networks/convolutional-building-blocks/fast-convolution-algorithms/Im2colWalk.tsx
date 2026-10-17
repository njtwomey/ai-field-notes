import { useMemo, useState } from 'react'
import { cn, Figure, Player, Readout, Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render'
import { tensor, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { im2col } from 'aifn-compute/nn/functional'

/** The worked example of the convolutional-layer note: a 4 × 4 image and a 3 × 3 vertical-edge kernel. */
const N = 4
const K = 3
const IMAGE = [1, 2, 0, 1, 0, 1, 3, 1, 2, 1, 0, 0, 1, 0, 1, 2]
const KERNEL = [1, 0, -1, 1, 0, -1, 1, 0, -1]
const M = N - K + 1

const highlight = 'bg-primary/15 font-semibold'

/**
 * im2col on the worked example, one output at a time: the window on the image, the column it becomes, and the dot
 * product of that column with the flattened kernel.
 */
export function Im2colWalk() {
  const [step, setStep] = useState(0)
  const columns = useMemo(() => toRows(im2col(tensor(IMAGE, [1, N, N]), K) as Tensor), [])
  const outputs = useMemo(
    () => columns[0].map((_, q) => columns.reduce((acc, row, r) => acc + KERNEL[r] * row[q], 0)),
    [columns],
  )
  const i = Math.floor(step / M)
  const j = step % M
  const inWindow = (r: number, c: number) => r >= i && r < i + K && c >= j && c < j + K
  const terms = columns.map((row, r) => `${KERNEL[r]}·${row[step]}`)

  return (
    <Figure
      title="im2col: windows become columns"
      caption="Step through the four outputs of a 3 × 3 kernel on a 4 × 4 image (no padding, stride 1). Each window of nine pixels (left, highlighted) is copied into one column of the 9 × 4 im2col matrix (middle), read row by row. The flattened kernel is one row of nine weights, and its product with column q is output q (right). Pixels in the middle of the image appear in up to four columns: the copy costs memory, and in exchange the whole layer becomes one matrix product."
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={M * M}
          label="output"
          format={(v) => `(${Math.floor(v / M)}, ${v % M})`}
        />
      }
      readouts={
        <>
          <Readout
            label={`column ${step} = window at (${i}, ${j})`}
            value={columns.map((row) => row[step]).join(', ')}
          />
          <Readout label={`y[${i}, ${j}] = kernel row · column`} value={`${terms.join(' + ')} = ${outputs[step]}`} />
          <Readout
            label="im2col size"
            value={`${K * K} × ${M * M} = ${K * K * M * M} numbers for a ${N * N}-pixel image`}
          />
        </>
      }
    >
      <div className="h-full overflow-auto">
        <div className="grid gap-4 md:grid-cols-[auto_minmax(0,1fr)_auto]">
          <div>
            <p className="mb-1 text-xs text-muted-foreground">image x</p>
            <Table className="w-auto font-mono text-xs">
              <TableBody>
                {Array.from({ length: N }, (_, r) => (
                  <TableRow key={r}>
                    {Array.from({ length: N }, (_, c) => (
                      <TableCell key={c} className={cn('px-2 text-center', inWindow(r, c) && highlight)}>
                        {IMAGE[r * N + c]}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">kernel row and im2col matrix (rows = taps (u, v))</p>
            <Table className="w-auto font-mono text-xs">
              <TableHeader>
                <TableRow>
                  <TableHead className="px-2">tap</TableHead>
                  <TableHead className="px-2 text-right">w</TableHead>
                  {Array.from({ length: M * M }, (_, q) => (
                    <TableHead key={q} className={cn('px-2 text-center', q === step && highlight)}>
                      col {q}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {columns.map((row, r) => (
                  <TableRow key={r}>
                    <TableCell className="px-2 text-muted-foreground">
                      ({Math.floor(r / K)}, {r % K})
                    </TableCell>
                    <TableCell className="px-2 text-right">{KERNEL[r]}</TableCell>
                    {row.map((v, q) => (
                      <TableCell key={q} className={cn('px-2 text-center', q === step && highlight)}>
                        {v}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">output y</p>
            <Table className="w-auto font-mono text-xs">
              <TableBody>
                {Array.from({ length: M }, (_, r) => (
                  <TableRow key={r}>
                    {Array.from({ length: M }, (_, c) => {
                      const q = r * M + c
                      return (
                        <TableCell key={c} className={cn('px-2 text-center', q === step && highlight)}>
                          {q <= step ? outputs[q] : '·'}
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </div>
    </Figure>
  )
}
