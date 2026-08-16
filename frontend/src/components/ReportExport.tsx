import { Download, Printer } from 'lucide-react'
import type { PredictResponse } from '../lib/types'

interface ReportExportProps {
  result: PredictResponse | null
}

export default function ReportExport({ result }: ReportExportProps) {
  const downloadJson = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `dr-grade-${result.grade_name.replace(/\s+/g, '-').toLowerCase()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="no-print flex items-center gap-3">
      <button
        type="button"
        onClick={downloadJson}
        disabled={!result}
        className="inline-flex items-center gap-2 rounded-xl border border-clinical-200 bg-white px-4 py-2.5 text-sm font-semibold text-clinical-700 shadow-sm transition hover:border-clinical-300 hover:bg-clinical-50 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Download size={15} /> JSON
      </button>
      <button
        type="button"
        onClick={() => window.print()}
        className="inline-flex items-center gap-2 rounded-xl bg-clinical-900 px-4 py-2.5 text-sm font-semibold text-white shadow-soft transition hover:bg-clinical-800 active:scale-[0.99]"
      >
        <Printer size={15} /> Export report
      </button>
    </div>
  )
}
