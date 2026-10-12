import React, { useState, useMemo } from 'react'

type TargetModality = 'boxes' | 'keypoints' | 'masks' | 'spans'

export const StructuredTargetExplorer: React.FC = () => {
  const [modality, setModality] = useState<TargetModality>('boxes')

  // Box state
  const [boxOffset, setBoxOffset] = useState<{ dx: number; dy: number; scaleW: number; scaleH: number }>({
    dx: 15,
    dy: -10,
    scaleW: 1.25,
    scaleH: 0.9,
  })

  // Keypoint state
  const [keypointSigma, setKeypointSigma] = useState<number>(3.5)
  const [keypointPos, setKeypointPos] = useState<{ x: number; y: number }>({ x: 120, y: 70 })

  // Anchor box constants
  const anchor = { x: 100, y: 80, w: 80, h: 60 }
  const groundTruthBox = {
    x: anchor.x + boxOffset.dx,
    y: anchor.y + boxOffset.dy,
    w: Math.round(anchor.w * boxOffset.scaleW),
    h: Math.round(anchor.h * boxOffset.scaleH),
  }

  // Regression targets: t_x, t_y, t_w, t_h
  const boxTargets = useMemo(() => {
    const tx = (groundTruthBox.x - anchor.x) / anchor.w
    const ty = (groundTruthBox.y - anchor.y) / anchor.h
    const tw = Math.log(groundTruthBox.w / anchor.w)
    const th = Math.log(groundTruthBox.h / anchor.h)
    return {
      tx: tx.toFixed(3),
      ty: ty.toFixed(3),
      tw: tw.toFixed(3),
      th: th.toFixed(3),
    }
  }, [anchor, groundTruthBox])

  // Span tokens
  const sampleTokens = [
    { text: 'Geoffrey', bio: 'B-PER', bilou: 'B-PER' },
    { text: 'Hinton', bio: 'I-PER', bilou: 'L-PER' },
    { text: 'won', bio: 'O', bilou: 'O' },
    { text: 'the', bio: 'O', bilou: 'O' },
    { text: 'Nobel', bio: 'B-MISC', bilou: 'B-MISC' },
    { text: 'Prize', bio: 'I-MISC', bilou: 'L-MISC' },
    { text: 'at', bio: 'O', bilou: 'O' },
    { text: 'Toronto', bio: 'B-LOC', bilou: 'U-LOC' },
  ]

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Structured Target Encodings Laboratory</h3>
        <p className="text-xs text-muted-foreground">
          Complex machine learning tasks cannot regress raw unstructured objects. Explore how models encode bounding
          boxes as anchor offsets, keypoints as Gaussian heatmaps, and entity spans as BIO token tags.
        </p>
      </div>

      {/* Modality Selector Tabs */}
      <div className="mb-5 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        {[
          { id: 'boxes', label: '1. Detection (Anchor Offsets)' },
          { id: 'keypoints', label: '2. Pose (Gaussian Heatmaps)' },
          { id: 'masks', label: '3. Segmentation (Pixel Masks)' },
          { id: 'spans', label: '4. NLP Spans (BIO Tagging)' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setModality(tab.id as TargetModality)}
            className={`rounded-lg border p-2.5 text-left transition-all ${
              modality === tab.id
                ? 'border-primary bg-primary font-semibold text-primary-foreground shadow-xs'
                : 'border-border bg-muted/40 text-muted-foreground hover:bg-muted'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Modality 1: Bounding Boxes */}
      {modality === 'boxes' && (
        <div>
          <div className="mb-4 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
            <div>
              <label className="mb-1 block font-medium text-foreground">
                X Offset (&Delta;x): <span className="font-mono text-primary">{boxOffset.dx}px</span>
              </label>
              <input
                type="range"
                min={-30}
                max={30}
                step={2}
                value={boxOffset.dx}
                onChange={(e) => setBoxOffset({ ...boxOffset, dx: Number(e.target.value) })}
                className="w-full accent-primary"
              />
            </div>

            <div>
              <label className="mb-1 block font-medium text-foreground">
                Y Offset (&Delta;y): <span className="font-mono text-primary">{boxOffset.dy}px</span>
              </label>
              <input
                type="range"
                min={-25}
                max={25}
                step={2}
                value={boxOffset.dy}
                onChange={(e) => setBoxOffset({ ...boxOffset, dy: Number(e.target.value) })}
                className="w-full accent-primary"
              />
            </div>

            <div>
              <label className="mb-1 block font-medium text-foreground">
                Width Scale: <span className="font-mono text-primary">{boxOffset.scaleW.toFixed(2)}x</span>
              </label>
              <input
                type="range"
                min={0.6}
                max={1.8}
                step={0.1}
                value={boxOffset.scaleW}
                onChange={(e) => setBoxOffset({ ...boxOffset, scaleW: Number(e.target.value) })}
                className="w-full accent-primary"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-lg border border-border bg-card/50 p-3">
              <div className="mb-2 flex items-center justify-between text-xs">
                <span className="font-semibold text-foreground">Spatial Anchor vs Ground Truth</span>
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="flex items-center gap-1">
                    <span className="inline-block h-0.5 w-3 border-t border-dashed border-amber-500"></span> Anchor Box
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block h-0.5 w-3 bg-emerald-500"></span> Ground Truth Box
                  </span>
                </div>
              </div>

              <svg viewBox="0 0 300 180" className="h-auto w-full rounded bg-muted/10">
                {/* Canvas grid */}
                <rect x="0" y="0" width="300" height="180" fill="none" stroke="currentColor" strokeOpacity={0.1} />

                {/* Predefined Anchor Box (dashed amber) */}
                <rect
                  x={anchor.x}
                  y={anchor.y}
                  width={anchor.w}
                  height={anchor.h}
                  fill="rgba(245, 158, 11, 0.1)"
                  stroke="#f59e0b"
                  strokeWidth="1.5"
                  strokeDasharray="4 3"
                />

                {/* Ground Truth Bounding Box (solid emerald) */}
                <rect
                  x={groundTruthBox.x}
                  y={groundTruthBox.y}
                  width={groundTruthBox.w}
                  height={groundTruthBox.h}
                  fill="rgba(16, 185, 129, 0.15)"
                  stroke="#10b981"
                  strokeWidth="2"
                />

                <circle cx={anchor.x} cy={anchor.y} r="3" fill="#f59e0b" />
                <circle cx={groundTruthBox.x} cy={groundTruthBox.y} r="3" fill="#10b981" />
              </svg>
            </div>

            <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4">
              <div>
                <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
                  Encoded Regression Target Vector
                </div>
                <div className="mb-3 grid grid-cols-2 gap-2 font-mono text-xs">
                  <div className="rounded border border-border bg-muted/20 p-2">
                    <span className="block font-sans text-[10px] text-muted-foreground">t_x = (x - x_a) / w_a</span>
                    <span className="text-base font-bold text-primary">{boxTargets.tx}</span>
                  </div>
                  <div className="rounded border border-border bg-muted/20 p-2">
                    <span className="block font-sans text-[10px] text-muted-foreground">t_y = (y - y_a) / h_a</span>
                    <span className="text-base font-bold text-primary">{boxTargets.ty}</span>
                  </div>
                  <div className="rounded border border-border bg-muted/20 p-2">
                    <span className="block font-sans text-[10px] text-muted-foreground">t_w = ln(w / w_a)</span>
                    <span className="text-base font-bold text-foreground">{boxTargets.tw}</span>
                  </div>
                  <div className="rounded border border-border bg-muted/20 p-2">
                    <span className="block font-sans text-[10px] text-muted-foreground">t_h = ln(h / h_a)</span>
                    <span className="text-base font-bold text-foreground">{boxTargets.th}</span>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  Log-space width/height encoding ensures that predicted dimensions are strictly positive upon
                  exponentiation: <em>w</em>
                  <sub>pred</sub> = <em>w</em>
                  <sub>a</sub> &middot; exp(<em>t</em>
                  <sub>w</sub>) &gt; 0.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modality 2: Keypoints */}
      {modality === 'keypoints' && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-border bg-card/50 p-3">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-semibold text-foreground">Click Canvas to Move Joint ($x_k, y_k$)</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                ({keypointPos.x}, {keypointPos.y})
              </span>
            </div>

            <svg
              viewBox="0 0 240 160"
              className="h-auto w-full cursor-crosshair rounded bg-muted/10"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const scaleX = 240 / rect.width
                const scaleY = 160 / rect.height
                setKeypointPos({
                  x: Math.round((e.clientX - rect.left) * scaleX),
                  y: Math.round((e.clientY - rect.top) * scaleY),
                })
              }}
            >
              {/* Gaussian Heatmap Circles */}
              {[3, 2, 1].map((ring) => (
                <circle
                  key={ring}
                  cx={keypointPos.x}
                  cy={keypointPos.y}
                  r={ring * keypointSigma * 2.2}
                  fill="#f43f5e"
                  fillOpacity={0.25 / ring}
                />
              ))}
              <circle cx={keypointPos.x} cy={keypointPos.y} r="3" fill="#f43f5e" />
            </svg>
          </div>

          <div className="flex flex-col justify-between rounded-lg border border-border bg-card/50 p-4 text-xs">
            <div>
              <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
                Continuous 2D Gaussian Target Encoding
              </div>
              <div className="mb-3">
                <label className="mb-1 block font-medium text-foreground">
                  Heatmap Spread (&sigma;): <span className="font-mono text-primary">{keypointSigma}px</span>
                </label>
                <input
                  type="range"
                  min={1.5}
                  max={6.0}
                  step={0.5}
                  value={keypointSigma}
                  onChange={(e) => setKeypointSigma(Number(e.target.value))}
                  className="w-full accent-primary"
                />
              </div>
              <code className="mb-2 block rounded bg-muted/50 p-2 font-mono text-[11px] text-primary">
                H(x, y) = exp(-((x - x_k)^2 + (y - y_k)^2) / (2&sigma;^2))
              </code>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Direct coordinate regression $(x_k, y_k)$ is brittle. Converting joints into smooth Gaussian heatmaps
                allows fully convolutional networks (U-Net, HRNet) to output spatial probability surfaces with sub-pixel
                centroid decoding.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Modality 3: Masks */}
      {modality === 'masks' && (
        <div className="rounded-lg border border-border bg-card/50 p-4 text-xs">
          <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
            Per-Pixel Classification Masks (Semantic Segmentation)
          </div>
          <p className="mb-3 text-[11px] leading-relaxed text-muted-foreground">
            Masks are encoded as 2D integer matrices <em>M</em> &isin; &#123;0, &hellip;, <em>C</em>-1&#125;
            <sup>
              <em>H</em> &times; <em>W</em>
            </sup>
            , where each coordinate (<em>i</em>, <em>j</em>) holds the categorical class index of that pixel (e.g.
            0=Background, 1=Road, 2=Vehicle, 3=Pedestrian).
          </p>
          <div className="grid grid-cols-1 gap-3 text-center font-mono sm:grid-cols-3">
            <div className="rounded border border-border bg-muted/20 p-3">
              <span className="block font-sans text-[10px] text-muted-foreground">Raw Label Matrix</span>
              <div className="mt-1 text-sm font-bold text-foreground">Shape: (H, W)</div>
              <span className="text-[9px] text-muted-foreground">uint8 class indices</span>
            </div>
            <div className="rounded border border-border bg-muted/20 p-3">
              <span className="block font-sans text-[10px] text-muted-foreground">One-Hot Softmax Target</span>
              <div className="mt-1 text-sm font-bold text-primary">Shape: (C, H, W)</div>
              <span className="text-[9px] text-muted-foreground">Binary volume tensor</span>
            </div>
            <div className="rounded border border-border bg-muted/20 p-3">
              <span className="block font-sans text-[10px] text-muted-foreground">Loss Functions</span>
              <div className="mt-1 text-sm font-bold text-emerald-500">CE + Soft Dice</div>
              <span className="text-[9px] text-muted-foreground">Intersection-over-Union</span>
            </div>
          </div>
        </div>
      )}

      {/* Modality 4: Spans */}
      {modality === 'spans' && (
        <div className="rounded-lg border border-border bg-card/50 p-4 text-xs">
          <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
            Sequential Named Entity Tagging (BIO vs BILOU Schemes)
          </div>
          <p className="mb-3 text-[11px] leading-relaxed text-muted-foreground">
            Continuous text spans cannot be regressed as character indices. Tokenizers align spans to discrete tokens
            via standard sequential tagging schemes:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-border font-sans text-[10px] text-muted-foreground uppercase">
                  <th className="px-2 py-1.5">Token</th>
                  <th className="px-2 py-1.5">BIO Tag</th>
                  <th className="px-2 py-1.5">BILOU Tag</th>
                  <th className="px-2 py-1.5 font-sans">Semantic Span Meaning</th>
                </tr>
              </thead>
              <tbody>
                {sampleTokens.map((t, idx) => (
                  <tr key={idx} className="border-b border-border/30">
                    <td className="px-2 py-2 font-bold text-foreground">{t.text}</td>
                    <td className="px-2 py-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] ${t.bio.startsWith('B') ? 'bg-blue-500/20 font-bold text-blue-500' : t.bio.startsWith('I') ? 'bg-amber-500/20 text-amber-500' : 'text-muted-foreground'}`}
                      >
                        {t.bio}
                      </span>
                    </td>
                    <td className="px-2 py-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] ${t.bilou.startsWith('U') ? 'bg-emerald-500/20 font-bold text-emerald-500' : 'text-muted-foreground'}`}
                      >
                        {t.bilou}
                      </span>
                    </td>
                    <td className="px-2 py-2 font-sans text-[11px] text-muted-foreground">
                      {t.bio.startsWith('B') && 'Begin new entity span'}
                      {t.bio.startsWith('I') && 'Continuation inside span'}
                      {t.bio === 'O' && 'Outside any entity'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
