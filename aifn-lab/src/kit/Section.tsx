import type { ReactNode } from 'react'

/** A titled block of the UI kit page. */
export function Section({
  title,
  description,
  children,
}: {
  title: string
  description?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="space-y-1 border-b pb-2">
        <h2 className="text-base font-semibold tracking-tight">{title}</h2>
        {description && <p className="max-w-prose text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  )
}
