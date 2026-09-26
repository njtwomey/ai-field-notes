/**
 * Reusable static figure from content/assets/: `<Asset name="transformer-architecture" caption="…" />`.
 * One file, many notes. Add the file once; reference it by name everywhere.
 */
const files = import.meta.glob<string>('@content/assets/*.{svg,png,jpg,webp}', {
  eager: true,
  query: '?url',
  import: 'default',
})

const byName = new Map(
  Object.entries(files).map(([p, url]) => [
    p
      .split('/')
      .pop()!
      .replace(/\.[^.]+$/, ''),
    url,
  ]),
)

export function Asset({ name, caption, alt, width }: { name: string; caption?: string; alt?: string; width?: number }) {
  const url = byName.get(name)
  if (!url) return <p className="text-destructive">Missing asset: content/assets/{name}.*</p>
  return (
    <figure className="not-prose my-8">
      <img src={url} alt={alt ?? caption ?? name} className="mx-auto rounded-md dark:invert-[.9]" style={{ width }} />
      {caption && <figcaption className="mt-2 text-center text-xs text-muted-foreground">{caption}</figcaption>}
    </figure>
  )
}
