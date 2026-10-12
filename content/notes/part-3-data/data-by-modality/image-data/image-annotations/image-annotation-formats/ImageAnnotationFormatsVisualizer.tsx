import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ImageAnnotationFormatsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'bbox_formats' | 'mask_compression' | 'keypoint_loss'>(
      [
        {
          value: 'bbox_formats',
          label: 'Bounding Box Conventions: VOC vs COCO vs YOLO Formats (Lin 2014, Redmon 2016)',
        },
        { value: 'mask_compression', label: 'Segmentation Masks: Dense Bitmap vs Polygon vs RLE Memory Footprint' },
        {
          value: 'keypoint_loss',
          label: 'Keypoint Annotations: Visibility Flags (v=0, 1, 2) vs Loss Gradient Masking',
        },
      ],
      'bbox_formats',
      { label: 'annotation format perspective' },
    ),
    canvasSize: slider(512, 2048, 1024, {
      step: 256,
      label: 'canvas image resolution W = H (pixels)',
    }),
    vertexCount: slider(6, 120, 30, {
      step: 6,
      label: 'polygon boundary vertices K',
    }),
  })

  const { visualizerMode, canvasSize, vertexCount } = state

  const results = useMemo(() => {
    if (visualizerMode === 'bbox_formats') {
      // 5 typical object box scales: Small, Medium, Large, Wide, Tall
      const boxScales = [1, 2, 3, 4, 5]
      // VOC: [xmin, ymin, xmax, ymax] absolute pixels
      // COCO: [xmin, ymin, w, h] absolute pixels
      // YOLO: [x_center, y_center, w, h] normalized in [0, 1]
      // Show normalized area % across the 5 scales
      const normalizedAreas = [4, 12, 35, 20, 25] // % of image
      const vocAbsoluteSizes = normalizedAreas.map((a) => Math.round((a / 100) * canvasSize))

      return {
        x: boxScales,
        y: vocAbsoluteSizes,
        baseline: normalizedAreas,
        stat1: `Canvas: ${canvasSize}x${canvasSize} px`,
        stat2: 'VOC: [xmin, ymin, xmax, ymax]',
        stat3: 'YOLO: [xc, yc, w, h] / Canvas',
        warning:
          'Converting between COCO and YOLO requires dividing by image width/height; omitting normalization breaks YOLO loss scales',
        axisXLabel: 'Object Scale Preset (1=Small, 2=Medium, 3=Large, 4=Wide, 5=Tall)',
        axisYLabel: 'Bounding Box Dimension (Pixels / Normalized %)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, Math.max(...vocAbsoluteSizes) * 1.15] as [number, number],
      }
    } else if (visualizerMode === 'mask_compression') {
      // Vertex counts K from 6 to 120
      const vertices = [6, 12, 24, 36, 48, 60, 72, 90, 120]
      // Dense uncompressed binary mask size: H * W / 8 bytes
      const denseMaskKb = (canvasSize * canvasSize) / (8 * 1024)

      // Polygon format size: K * 2 floats * 4 bytes = 8 * K bytes -> in KB
      const polygonKb = vertices.map((k) => (k * 8) / 1024)
      // Compressed COCO Run-Length Encoding (RLE): typically 1 KB to 5 KB depending on perimeter
      const rleKb = vertices.map((k) => (k * 40) / 1024 + 1.2)

      const currPolygonKb = (vertexCount * 8) / 1024
      const currRLEKb = (vertexCount * 40) / 1024 + 1.2

      return {
        x: vertices,
        y: rleKb,
        baseline: polygonKb,
        stat1: `Dense bitmap: ${denseMaskKb.toFixed(0)} KB`,
        stat2: `Polygon (${vertexCount} pts): ${currPolygonKb.toFixed(2)} KB`,
        stat3: `COCO RLE: ${currRLEKb.toFixed(2)} KB`,
        warning:
          'Dense bitmaps waste megabytes per image; COCO polygons compress simple shapes, while RLE encodes complex amorphous masks efficiently',
        axisXLabel: 'Segmentation Polygon Vertex Complexity (Points K)',
        axisYLabel: 'Annotation Memory Footprint (KB per Instance)',
        xRange: [4, 125] as [number, number],
        yRange: [0, Math.max(...rleKb) * 1.25] as [number, number],
      }
    } else {
      // Keypoint visibility categories: v=0 (not labeled/unobserved), v=1 (labeled but occluded), v=2 (visible)
      const visibilities = [0, 1, 2]
      // Loss weight assigned: v=0 -> 0.0 (masked out), v=1 -> 1.0 (supervised heatmap), v=2 -> 1.0
      const lossWeights = [0.0, 1.0, 1.0]
      const detectionConfidence = [0.0, 0.45, 0.95]

      return {
        x: visibilities,
        y: detectionConfidence.map((c) => c * 100),
        baseline: lossWeights.map((w) => w * 100),
        stat1: 'COCO Keypoint Standard (17 joints)',
        stat2: 'Flags: v=0 (skip), v=1 (occluded), v=2 (visible)',
        stat3: 'Loss Masking Active',
        warning:
          'Failing to mask out unannotated keypoints (v=0) penalizes models for predicting genuine anatomical joints, corrupting training',
        axisXLabel: 'Keypoint Visibility Code (0=Invisible, 1=Occluded, 2=Visible)',
        axisYLabel: 'Loss Weight / Confidence Benchmark (%)',
        xRange: [-0.5, 2.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, canvasSize, vertexCount])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Image Annotations: Bounding Box Coordinate Standards, Mask Encoding, and Keypoints"
      state={state}
      hoverReadout={false}
      caption="Common computer vision annotation formats: Discrepancies between VOC, COCO, and YOLO bounding box coordinate systems, compression efficiency of polygons and RLE, and keypoint loss masking."
      readouts={
        <>
          <Readout label="Format Setup" value={results.stat1} />
          <Readout label="Coordinate Rule" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'bbox_formats'
                ? 'Target Format'
                : visualizerMode === 'mask_compression'
                  ? 'RLE Footprint'
                  : 'Supervision Rule'
            }
            value={results.stat3}
            color={visualizerMode === 'mask_compression' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Engineering Warning" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Annotation Format Transformation Profile">
        <Curve
          name={
            visualizerMode === 'bbox_formats'
              ? 'Normalized Bounding Box Area (%)'
              : visualizerMode === 'mask_compression'
                ? 'Polygon Vector Representation Footprint (KB)'
                : 'Loss Supervision Mask Weight (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'bbox_formats'
              ? 'Absolute Pixel Dimension (VOC Format)'
              : visualizerMode === 'mask_compression'
                ? 'Run-Length Encoded (RLE) Compressed Footprint (KB)'
                : 'Expected Keypoint Heatmap Confidence (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Format Presets" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
