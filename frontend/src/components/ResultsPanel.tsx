import { motion } from 'motion/react'
import { ArrowRight, Gauge, Timer, Wand2 } from 'lucide-react'
import type { PredictResponse } from '../lib/types'
import { GRADE_NAMES } from '../lib/types'
import { GRADE_STYLES, SEVERITY_LABEL } from '../lib/gradeStyles'
import CountUp from './CountUp'
import ProbabilityBars from './ProbabilityBars'

interface ResultsPanelProps {
  result: PredictResponse
  onReset: () => void
}

export default function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  const style = GRADE_STYLES[result.grade]

  return (
    <motion.section
      id="results"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="print-block grid gap-6 lg:grid-cols-5"
    >
      {/* Left: headline result */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="lg:col-span-2 rounded-3xl border border-clinical-100 bg-white p-7 shadow-card"
      >
        <p className="text-xs font-bold uppercase tracking-widest text-clinical-400">
          Diagnosis result
        </p>

        <div className="mt-5 flex items-end gap-4">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 180, damping: 14, delay: 0.25 }}
            className={`flex h-20 w-20 shrink-0 items-center justify-center rounded-2xl text-4xl font-extrabold ring-8 ${style.bg} ${style.ring} ${style.text}`}
          >
            {result.grade}
          </motion.div>
          <div className="min-w-0">
            <h3 className={`font-display text-2xl font-bold leading-tight ${style.text}`}>
              {result.grade_name}
            </h3>
            <p className="text-sm font-medium text-clinical-500">
              {SEVERITY_LABEL[result.grade]} severity level
            </p>
          </div>
        </div>

        <div className="mt-6 flex items-center gap-3">
          <div className="flex-1">
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium text-clinical-500">Confidence</span>
              <span className={`font-mono text-lg font-bold tabular-nums ${style.text}`}>
                <CountUp value={result.confidence * 100} decimals={1} />%
              </span>
            </div>
            <div className="mt-2 h-2.5 overflow-hidden rounded-full bg-clinical-100">
              <motion.div
                className={`h-full rounded-full ${style.bar}`}
                initial={{ width: 0 }}
                animate={{ width: `${result.confidence * 100}%` }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay: 0.35 }}
              />
            </div>
          </div>
          <div className="ml-3 flex shrink-0 items-center gap-1.5 rounded-full bg-clinical-50 px-3 py-1.5 text-xs font-medium text-clinical-600">
            <Timer size={13} />
            {result.latency_ms >= 1000 ? `${(result.latency_ms / 1000).toFixed(1)}s` : `${Math.round(result.latency_ms)}ms`}
          </div>
        </div>

        <div className="mt-6 space-y-2.5 rounded-2xl bg-clinical-50 p-4 text-sm">
          <p className="flex items-start gap-2 text-clinical-700">
            <Wand2 size={15} className="mt-0.5 shrink-0 text-accent-600" />
            {result.grade_description}
          </p>
          <p className="flex items-center gap-2 text-xs text-clinical-500">
            <Gauge size={13} className="shrink-0 text-clinical-400" />
            {GRADE_NAMES[result.grade]} · image analyzed by EfficientNet-B3 (CORAL ordinal)
            {result.tta ? ' · TTA on' : ' · TTA off'}
          </p>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-clinical-900 px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:bg-clinical-800 active:scale-[0.99]"
        >
          Analyze another image <ArrowRight size={16} />
        </button>
      </motion.div>

      {/* Right: probability distribution */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="lg:col-span-3 rounded-3xl border border-clinical-100 bg-white p-7 shadow-card"
      >
        <div className="mb-5 flex items-center justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-widest text-clinical-400">
              Probability distribution
            </p>
            <p className="mt-1 text-sm text-clinical-500">
              Predicted severity across all five DR grades
            </p>
          </div>
          <div
            className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-wide ${style.bg} ${style.text} ring-1 ${style.ring}`}
          >
            Grade {result.grade} · {SEVERITY_LABEL[result.grade]}
          </div>
        </div>
        <ProbabilityBars probs={result.probs} predictedGrade={result.grade} />
        <p className="mt-5 text-xs text-clinical-400">
          Grades are ordinal (0 → 4): the model optimizes cumulative thresholds so off-by-many
          mistakes are penalised harder than adjacent-grade confusions.
        </p>
      </motion.div>
    </motion.section>
  )
}
