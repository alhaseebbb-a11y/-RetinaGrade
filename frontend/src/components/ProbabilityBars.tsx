import { motion } from 'motion/react'
import { GRADE_NAMES } from '../lib/types'
import { GRADE_STYLES } from '../lib/gradeStyles'

interface ProbabilityBarsProps {
  probs: Record<string, number>
  predictedGrade: number
}

export default function ProbabilityBars({ probs, predictedGrade }: ProbabilityBarsProps) {
  const rows = GRADE_NAMES.map((name, i) => ({
    grade: i,
    name,
    prob: probs[String(i)] ?? 0,
    style: GRADE_STYLES[i],
    isPred: i === predictedGrade,
  })).sort((a, b) => b.prob - a.prob)

  return (
    <div className="space-y-3">
      {rows.map((row, idx) => (
        <div key={row.grade} className="group">
          <div className="mb-1 flex items-baseline justify-between text-sm">
            <span
              className={`flex items-center gap-2 font-medium ${row.isPred ? row.style.text : 'text-clinical-700'}`}
            >
              <span
                className={`inline-block h-2 w-2 rounded-full ${row.isPred ? row.style.dot : 'bg-clinical-200'}`}
              />
              Grade {row.grade} · {row.name}
              {row.isPred && (
                <span className="rounded-full bg-clinical-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-clinical-600">
                  Predicted
                </span>
              )}
            </span>
            <span className={`font-mono text-xs tabular-nums ${row.isPred ? row.style.text : 'text-clinical-400'}`}>
              {(row.prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-clinical-100">
            <motion.div
              className={`h-full rounded-full ${row.isPred ? row.style.bar : 'bg-clinical-300'}`}
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(row.prob * 100, 1)}%` }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.15 + idx * 0.08 }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
