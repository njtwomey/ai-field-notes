/** Numbers with six significant figures; integers exactly. */
export function formatValue(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return String(v)
  if (!Number.isFinite(v)) return String(v)
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(4)
  return String(Number(v.toPrecision(6)))
}
