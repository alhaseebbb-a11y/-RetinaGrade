import type { GradeStyle } from './types'

export const GRADE_STYLES: GradeStyle[] = [
  { text: 'text-emerald-700', bg: 'bg-emerald-50', ring: 'ring-emerald-200', bar: 'bg-emerald-500', dot: 'bg-emerald-500', hex: '#10b981' },
  { text: 'text-lime-700', bg: 'bg-lime-50', ring: 'ring-lime-200', bar: 'bg-lime-500', dot: 'bg-lime-500', hex: '#84cc16' },
  { text: 'text-amber-700', bg: 'bg-amber-50', ring: 'ring-amber-200', bar: 'bg-amber-500', dot: 'bg-amber-500', hex: '#f59e0b' },
  { text: 'text-orange-700', bg: 'bg-orange-50', ring: 'ring-orange-200', bar: 'bg-orange-500', dot: 'bg-orange-500', hex: '#f97316' },
  { text: 'text-red-700', bg: 'bg-red-50', ring: 'ring-red-200', bar: 'bg-red-500', dot: 'bg-red-500', hex: '#ef4444' },
]

export const SEVERITY_LABEL = ['Healthy', 'Mild', 'Moderate', 'Severe', 'Proliferative'] as const
