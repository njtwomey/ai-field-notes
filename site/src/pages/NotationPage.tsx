import katex from 'katex'
import { macroExample, macroGroups, macros } from '@content/macros'
import { PageContainer } from '@/components/layout/AppShell'

function render(tex: string): string {
  return katex.renderToString(tex, { macros: { ...macros }, throwOnError: false, strict: 'ignore' })
}

/** Every maths macro available in notes, from content/macros.ts, with its rendering. */
export default function NotationPage() {
  return (
    <PageContainer>
      <h1 className="font-prose text-4xl font-bold">Notation</h1>
      <p className="mt-3 max-w-2xl font-prose text-lg text-muted-foreground">
        Bold lowercase letters are vectors and bold capitals are matrices. Calligraphic capitals are sets or families.
        Every note uses these macros, defined once in <code className="font-mono text-sm">content/macros.ts</code>.
      </p>
      <div className="mt-10 space-y-12">
        {macroGroups.map((group) => (
          <section key={group.title} id={group.title.toLowerCase().replaceAll(' ', '-')} className="space-y-3">
            <h2 className="font-prose text-2xl font-bold">{group.title}</h2>
            {group.description && <p className="text-sm text-muted-foreground">{group.description}</p>}
            <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
              {Object.entries(group.macros).map(([name, body]) => (
                <li key={name} className="flex flex-col gap-1 rounded-md border px-3 py-2" title={body}>
                  <div className="flex items-baseline justify-between gap-2">
                    <code className="truncate font-mono text-xs">
                      {group.examples?.[name] ? name : macroExample(group, name)}
                    </code>
                    <span
                      className="shrink-0 font-prose text-lg"
                      dangerouslySetInnerHTML={{ __html: render(macroExample(group, name)) }}
                    />
                  </div>
                  <code className="truncate font-mono text-[10px] text-muted-foreground">{body}</code>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </PageContainer>
  )
}
