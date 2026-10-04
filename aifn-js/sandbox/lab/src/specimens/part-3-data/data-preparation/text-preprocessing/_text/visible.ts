/** A token as shown on a chip: spaces, tabs and newlines made visible. */
export const visible = (t: string) => t.replaceAll(' ', '␣').replaceAll('\t', '⇥').replaceAll('\n', '↵')
