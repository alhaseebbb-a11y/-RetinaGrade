export const GRADE_NAMES = [
  'No DR',
  'Mild',
  'Moderate',
  'Severe',
  'Proliferative DR',
] as const

export const GRADE_DESCRIPTIONS = [
  'No signs of diabetic retinopathy.',
  'Microaneurysms only — earliest visible change.',
  'More than just microaneurysms but less than severe.',
  'Extensive hemorrhages, venous beading, IRMA.',
  'Neovascularization, vitreous/pre-retinal hemorrhage.',
] as const

export interface ModelMeta {
  path: string
  size_mb: number
  image_size: number
  num_classes: number
  device: string
}

export interface HealthResponse {
  status: 'ok' | 'loading'
  model: ModelMeta
}

export interface PredictResponse {
  grade: number
  grade_name: string
  grade_description: string
  confidence: number
  probs: Record<string, number>
  threshold_probs: Record<string, number>
  tta: boolean
  latency_ms: number
  filename: string
  model: ModelMeta
}

export interface PerClassMetrics {
  per_class_precision: Record<string, number>
  per_class_recall: Record<string, number>
  per_class_f1: Record<string, number>
}

export interface GradeStyle {
  text: string
  bg: string
  ring: string
  bar: string
  dot: string
  hex: string
}

export interface MetricsResponse extends PerClassMetrics {
  test_accuracy: number
  qwk: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  weighted_f1: number
  tta: boolean
  n_test: number
}
