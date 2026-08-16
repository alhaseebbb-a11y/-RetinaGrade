import { motion } from 'motion/react'
import { Database, PieChart } from 'lucide-react'
import type { MetricsResponse } from '../lib/types'
import { GRADE_NAMES } from '../lib/types'
import { GRADE_STYLES } from '../lib/gradeStyles'
import CountUp from './CountUp'

interface ModelDashboardProps {
  metrics: MetricsResponse | null
  loading: boolean
}

export default function ModelDashboard({ metrics, loading }: ModelDashboardProps) {
  return (
    <motion.section
      id="dashboard"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="no-print mt-16"
    >
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-clinical-500 to-accent-500 shadow-soft">
          <PieChart className="text-white" size={17} />
        </div>
        <div>
          <h2 className="font-display text-xl font-bold text-clinical-900">Model performance</h2>
          <p className="text-sm text-clinical-500">
            Held-out test set · {metrics ? metrics.n_test.toLocaleString() : '…'} fundus images
          </p>
        </div>
      </div>

      {loading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-32 rounded-3xl" />
          ))}
        </div>
      )}

      {metrics && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: 'Quadratic Weighted Kappa', value: metrics.qwk, decimals: 4, pct: false, accent: true },
              { label: 'Accuracy', value: metrics.test_accuracy, decimals: 4, pct: true },
              { label: 'Macro F1', value: metrics.macro_f1, decimals: 4, pct: false },
              { label: 'Weighted F1', value: metrics.weighted_f1, decimals: 4, pct: false },
            ].map((card, idx) => (
              <motion.div
                key={card.label}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: idx * 0.07 }}
                className={`rounded-3xl border p-5 shadow-card ${
                  card.accent ? 'border-accent-400/30 bg-gradient-to-br from-white to-accent-400/10' : 'border-clinical-100 bg-white'
                }`}
              >
                <p className="text-[11px] font-bold uppercase tracking-widest text-clinical-400">
                  {card.label}
                </p>
                <p
                  className={`mt-2 font-mono text-3xl font-bold tabular-nums ${
                    card.accent ? 'text-accent-600' : 'text-clinical-900'
                  }`}
                >
                  <CountUp value={card.decimals > 0 ? card.value : card.value} decimals={card.decimals} />
                  {card.pct ? '%' : ''}
                </p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="mt-4 rounded-3xl border border-clinical-100 bg-white p-6 shadow-card"
          >
            <div className="mb-4 flex items-center gap-2 text-sm font-medium text-clinical-600">
              <Database size={15} className="text-accent-600" />
              Per-class F1 on the test set
            </div>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
              {GRADE_NAMES.map((name, i) => {
                const f1 = metrics.per_class_f1?.[String(i)] ?? 0
                const style = GRADE_STYLES[i]
                return (
                  <div key={i} className="rounded-2xl bg-clinical-50/70 p-4">
                    <div className="flex items-center gap-2">
                      <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                      <span className="text-xs font-semibold text-clinical-700">
                        Grade {i}
                      </span>
                    </div>
                    <p className="mt-1 truncate text-[11px] text-clinical-400">{name}</p>
                    <p className={`mt-2 font-mono text-xl font-bold tabular-nums ${style.text}`}>
                      <CountUp value={f1} decimals={3} />
                    </p>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-clinical-100">
                      <motion.div
                        className={`h-full rounded-full ${style.bar}`}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${f1 * 100}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        </>
      )}

      {!loading && !metrics && (
        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Metrics unavailable — backend offline.
        </p>
      )}
    </motion.section>
  )
}
