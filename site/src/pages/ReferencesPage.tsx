import { Link } from 'react-router'
import { ReferenceCard } from '@/components/content/ReferenceCard'
import { PageContainer } from '@/components/layout/AppShell'
import { notes, noteUrl, references } from '@/lib/content'

/** Every source in content/references.yaml, with the notes that cite it. */
export function ReferencesPage() {
  const entries = Object.entries(references).sort(([, a], [, b]) => (b.year ?? 0) - (a.year ?? 0))
  return (
    <PageContainer>
      <h1 className="mb-8 font-prose text-3xl font-bold">References</h1>
      <ul className="divide-y rounded-lg border">
        {entries.map(([key, ref]) => {
          const citing = notes.filter((n) => n.cited.includes(key) || n.references.includes(key))
          return (
            <li key={key} id={key} className="space-y-2 px-4 py-3">
              <ReferenceCard reference={ref} />
              {citing.length > 0 && (
                <p className="pl-5.5 text-xs text-muted-foreground">
                  Cited in{' '}
                  {citing.map((n, i) => (
                    <span key={n.slug}>
                      {i > 0 && ', '}
                      <Link to={noteUrl(n.slug)} className="underline">
                        {n.title}
                      </Link>
                    </span>
                  ))}
                </p>
              )}
            </li>
          )
        })}
      </ul>
    </PageContainer>
  )
}
