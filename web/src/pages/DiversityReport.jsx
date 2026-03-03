import { useState, useRef, useCallback, Fragment } from 'react';
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const DATASET_COLORS = ['#3949ab', '#e53935', '#2e7d32', '#f57c00', '#7b1fa2', '#00838f'];

// ── Metric metadata ────────────────────────────────────────────────────────

const DIVERSITY_METRICS = [
  { key: 'subject_words',        label: 'Subject Words',          short: 'Subjects' },
  { key: 'invariant_templates',  label: 'Invariant Templates',    short: 'Invariants' },
  { key: 'assert_templates',     label: 'Assert Templates',       short: 'Asserts' },
  { key: 'requires_templates',   label: 'Requires Templates',     short: 'Requires' },
  { key: 'ensures_templates',    label: 'Ensures Templates',      short: 'Ensures' },
  { key: 'loop_skeletons',       label: 'Loop Skeletons',         short: 'Loops' },
  { key: 'quantifier_signatures',label: 'Quantifier Signatures',  short: 'Quantifiers' },
];

const COMPLEXITY_METRICS = [
  { key: 'body_size',               label: 'Body Size (lines)',          short: 'Body size' },
  { key: 'n_loops',                 label: '# Loops per Method',         short: '# Loops' },
  { key: 'max_loop_depth',          label: 'Max Loop Depth per Method',  short: 'Max depth' },
  { key: 'n_idents_in_asserts',     label: 'Identifiers in Asserts',     short: 'Idents/assert' },
  { key: 'n_idents_in_invs',        label: 'Identifiers in Invariants',  short: 'Idents/inv' },
  { key: 'n_quantifiers_per_clause',label: 'Quantifiers per Clause',     short: 'Quants/clause' },
];

// ── Data processing ────────────────────────────────────────────────────────

function buildReport(files) {
  return files.map((f, i) => ({
    id: f.id,
    label: f.label,
    color: DATASET_COLORS[i % DATASET_COLORS.length],
    data: f.data,
    n_programs: f.data.n_programs ?? 0,
    source: f.data.source ?? '',
  }));
}

/** Build data array for the entropy comparison chart (one section at a time). */
function entropyChartData(datasets, metrics) {
  return metrics.map(m => {
    const row = { metric: m.short };
    for (const ds of datasets) {
      const section = ds.data.diversity ?? ds.data.complexity ?? {};
      row[ds.label] = (section[m.key]?.entropy ?? 0).toFixed(3);
    }
    return row;
  });
}

/**
 * Build a unified top-k table for a given metric across all datasets.
 * Returns rows sorted by total count desc.
 */
function topKTableData(datasets, section, metricKey) {
  const totals = new Map(); // value -> {[label]: count}
  for (const ds of datasets) {
    const topK = ds.data[section]?.[metricKey]?.top_k ?? [];
    for (const { value, count } of topK) {
      if (!totals.has(value)) totals.set(value, {});
      totals.get(value)[ds.label] = count;
    }
  }
  const rows = [...totals.entries()].map(([value, counts]) => ({
    value,
    counts,
    total: Object.values(counts).reduce((a, b) => a + b, 0),
  }));
  rows.sort((a, b) => b.total - a.total);
  return rows;
}

/**
 * Build distribution chart data for a complexity metric.
 * Merges top_k from all datasets, sorts by numeric value.
 * When relative=true, counts are divided by each dataset's n_programs.
 */
function distributionChartData(datasets, metricKey, relative) {
  const valueSet = new Map(); // value -> {[label]: count}
  for (const ds of datasets) {
    const metric = ds.data.complexity?.[metricKey] ?? {};
    const topK = metric.top_k ?? [];
    const total = relative ? (metric.n_total || 1) : 1;
    for (const { value, count } of topK) {
      if (!valueSet.has(value)) valueSet.set(value, {});
      valueSet.get(value)[ds.label] = count / total;
    }
  }
  const rows = [...valueSet.entries()].map(([value, counts]) => ({
    value: String(value),
    numValue: Number(value),
    ...counts,
  }));
  rows.sort((a, b) => a.numValue - b.numValue);
  return rows;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function SummaryTable({ datasets }) {
  return (
    <table className="stat-table" style={{ width: '100%' }}>
      <thead>
        <tr>
          <th>Dataset</th>
          <th>Programs</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {datasets.map(ds => (
          <tr key={ds.id}>
            <td>
              <span className="color-dot" style={{ background: ds.color }} />
              <strong>{ds.label}</strong>
            </td>
            <td>{ds.n_programs.toLocaleString()}</td>
            <td className="muted small" style={{ fontFamily: 'monospace', fontSize: '.75rem' }}>{ds.source}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EntropyChart({ datasets, metrics, title }) {
  const chartData = entropyChartData(datasets, metrics);
  return (
    <div>
      {title && <h3 style={{ fontSize: '.875rem', fontWeight: 600, color: '#555', marginBottom: '.75rem' }}>{title}</h3>}
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ bottom: 8, left: 4, right: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
          <YAxis
            label={{ value: 'Entropy (bits)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10 }}
            tick={{ fontSize: 11 }}
            width={72}
          />
          <Tooltip formatter={(v) => [Number(v).toFixed(3) + ' bits']} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {datasets.map(ds => (
            <Bar key={ds.label} dataKey={ds.label} fill={ds.color} radius={[3, 3, 0, 0]} maxBarSize={40} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TopKTable({ datasets, section, metricKey, metricLabel }) {
  const rows = topKTableData(datasets, section, metricKey);
  if (rows.length === 0) {
    return <p className="muted small">No data.</p>;
  }

  const maxCount = Math.max(...rows.flatMap(r => Object.values(r.counts)));

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="stat-table" style={{ width: '100%', tableLayout: 'auto' }}>
        <thead>
          <tr>
            <th style={{ minWidth: 160 }}>Value ({metricLabel})</th>
            {datasets.map(ds => (
              <th key={ds.id} style={{ minWidth: 110 }}>
                <span className="color-dot" style={{ background: ds.color }} />
                {ds.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ value, counts }) => (
            <tr key={value}>
              <td style={{ fontFamily: 'monospace', fontSize: '.78rem', maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={value}>
                {value}
              </td>
              {datasets.map(ds => {
                const count = counts[ds.label] ?? 0;
                const pct = maxCount > 0 ? count / maxCount : 0;
                return (
                  <td key={ds.id}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{
                        flex: 1,
                        height: 8,
                        borderRadius: 4,
                        background: count > 0 ? ds.color : '#eee',
                        opacity: count > 0 ? 0.2 + pct * 0.8 : 1,
                        width: `${Math.round(pct * 100)}%`,
                        minWidth: count > 0 ? 4 : 0,
                      }} />
                      <span style={{ fontSize: '.78rem', color: count > 0 ? '#333' : '#bbb', whiteSpace: 'nowrap' }}>
                        {count > 0 ? count.toLocaleString() : '—'}
                      </span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComplexitySummaryTable({ datasets }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="stat-table" style={{ width: '100%' }}>
        <thead>
          <tr>
            <th>Metric</th>
            {datasets.map(ds => (
              <th key={ds.id} colSpan={3}>
                <span className="color-dot" style={{ background: ds.color }} />
                {ds.label}
              </th>
            ))}
          </tr>
          <tr>
            <th />
            {datasets.map(ds => (
              <Fragment key={ds.id}>
                <th style={{ color: '#aaa', fontWeight: 500, fontSize: '.78rem' }}>entropy</th>
                <th style={{ color: '#aaa', fontWeight: 500, fontSize: '.78rem' }}>mean</th>
                <th style={{ color: '#aaa', fontWeight: 500, fontSize: '.78rem' }}>median</th>
              </Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {COMPLEXITY_METRICS.map(m => (
            <tr key={m.key}>
              <td style={{ fontWeight: 500 }}>{m.label}</td>
              {datasets.map(ds => {
                const d = ds.data.complexity?.[m.key] ?? {};
                return (
                  <Fragment key={ds.id}>
                    <td>{d.entropy != null ? d.entropy.toFixed(3) : '—'}</td>
                    <td>{d.mean != null ? d.mean.toFixed(2) : '—'}</td>
                    <td>{d.median ?? '—'}</td>
                  </Fragment>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DistributionChart({ datasets, metricKey, relative }) {
  const chartData = distributionChartData(datasets, metricKey, relative);
  if (chartData.length === 0) return <p className="muted small">No data.</p>;

  const yFormatter = relative
    ? v => `${(v * 100).toFixed(3)}%`
    : v => v.toLocaleString();

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ bottom: 8, left: 4, right: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="value" tick={{ fontSize: 11 }} label={{ value: 'Value', position: 'insideBottom', offset: -4, fontSize: 10 }} />
        <YAxis tickFormatter={yFormatter} tick={{ fontSize: 11 }} width={56} />
        <Tooltip formatter={(v, name) => [yFormatter(v), name]} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {datasets.map(ds => (
          <Line
            key={ds.label}
            type="monotone"
            dataKey={ds.label}
            stroke={ds.color}
            strokeWidth={2}
            dot={{ r: 3 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Metric picker ──────────────────────────────────────────────────────────

function MetricPicker({ metrics, selected, onChange }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem', marginBottom: '1rem' }}>
      {metrics.map(m => (
        <button
          key={m.key}
          onClick={() => onChange(m.key)}
          style={{
            padding: '4px 12px',
            borderRadius: 20,
            border: '2px solid',
            borderColor: selected === m.key ? '#3949ab' : '#ddd',
            background: selected === m.key ? '#3949ab' : '#fff',
            color: selected === m.key ? '#fff' : '#555',
            fontSize: '.8rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all .15s',
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function DiversityReport() {
  const [files, setFiles] = useState([]);
  const [datasets, setDatasets] = useState(null);
  const [selectedDiversityMetric, setSelectedDiversityMetric] = useState(DIVERSITY_METRICS[0].key);
  const [selectedComplexityMetric, setSelectedComplexityMetric] = useState(COMPLEXITY_METRICS[0].key);
  const [distributionRelative, setDistributionRelative] = useState(false);
  const fileInputRef = useRef(null);

  const addFiles = useCallback(async (fileList) => {
    const incoming = [];
    for (const file of Array.from(fileList)) {
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        // Basic validation
        if (!data.diversity && !data.complexity) {
          alert(`${file.name} doesn't look like a diversity report (missing diversity/complexity keys).`);
          continue;
        }
        incoming.push({
          id: Math.random().toString(36).slice(2),
          label: data.name ?? file.name.replace(/\.json$/, ''),
          data,
        });
      } catch (e) {
        alert(`Failed to parse ${file.name}: ${e.message}`);
      }
    }
    setFiles(prev => [...prev, ...incoming]);
    setDatasets(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const removeFile = (id) => { setFiles(prev => prev.filter(f => f.id !== id)); setDatasets(null); };
  const updateLabel = (id, label) => { setFiles(prev => prev.map(f => f.id === id ? { ...f, label } : f)); setDatasets(null); };

  const handleBuild = () => {
    if (!files.length) return;
    setDatasets(buildReport(files));
  };

  const selDiv = DIVERSITY_METRICS.find(m => m.key === selectedDiversityMetric);
  const selCmplx = COMPLEXITY_METRICS.find(m => m.key === selectedComplexityMetric);

  return (
    <div className="page">
      <h1>Diversity Report</h1>

      {/* ── File loading ── */}
      <section className="section">
        <h2>Load Diversity Report Files</h2>
        <p className="muted small" style={{ marginBottom: '1rem' }}>
          Drop one or more <code>diversity-*.json</code> files produced by <code>diversity.py</code>.
          Multiple files are compared side by side.
        </p>
        <div
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
        >
          Drag &amp; drop diversity report JSON files here, or click to browse
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            multiple
            style={{ display: 'none' }}
            onChange={e => { addFiles(e.target.files); e.target.value = ''; }}
          />
        </div>

        {files.length > 0 && (
          <table className="file-table">
            <thead>
              <tr><th>Label</th><th>Programs</th><th></th></tr>
            </thead>
            <tbody>
              {files.map(f => (
                <tr key={f.id}>
                  <td>
                    <input
                      className="label-input"
                      value={f.label}
                      onChange={e => updateLabel(f.id, e.target.value)}
                    />
                  </td>
                  <td className="muted">{(f.data.n_programs ?? '?').toLocaleString()}</td>
                  <td><button className="btn-remove" onClick={() => removeFile(f.id)}>✕</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <button className="btn-primary" onClick={handleBuild} disabled={!files.length}>
          Build Report
        </button>
      </section>

      {datasets && (
        <>
          {/* ── Dataset summary ── */}
          <section className="section">
            <h2>Datasets</h2>
            <SummaryTable datasets={datasets} />
          </section>

          {/* ── Diversity metrics ── */}
          <section className="section">
            <h2>Diversity</h2>
            <p className="muted small" style={{ marginBottom: '1rem' }}>
              Shannon entropy (bits) of each categorical distribution. Higher = more diverse.
            </p>

            <EntropyChart datasets={datasets} metrics={DIVERSITY_METRICS} />

            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: '.875rem', fontWeight: 600, color: '#555', marginBottom: '.75rem' }}>
                Top-k Most Frequent Values
              </h3>
              <MetricPicker metrics={DIVERSITY_METRICS} selected={selectedDiversityMetric} onChange={setSelectedDiversityMetric} />
              <TopKTable
                datasets={datasets}
                section="diversity"
                metricKey={selectedDiversityMetric}
                metricLabel={selDiv?.label ?? selectedDiversityMetric}
              />
            </div>
          </section>

          {/* ── Complexity metrics ── */}
          <section className="section">
            <h2>Complexity</h2>
            <p className="muted small" style={{ marginBottom: '1rem' }}>
              Distribution of structural complexity measures across programs.
            </p>

            <ComplexitySummaryTable datasets={datasets} />

            <div style={{ marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.75rem' }}>
                <h3 style={{ fontSize: '.875rem', fontWeight: 600, color: '#555' }}>
                  Value Distribution
                </h3>
                <button
                  onClick={() => setDistributionRelative(r => !r)}
                  style={{
                    padding: '3px 12px',
                    borderRadius: 20,
                    border: '2px solid',
                    borderColor: distributionRelative ? '#3949ab' : '#ddd',
                    background: distributionRelative ? '#eef0ff' : '#fff',
                    color: distributionRelative ? '#3949ab' : '#888',
                    fontSize: '.78rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all .15s',
                  }}
                >
                  {distributionRelative ? 'distribution' : 'raw count'}
                </button>
              </div>
              <MetricPicker metrics={COMPLEXITY_METRICS} selected={selectedComplexityMetric} onChange={setSelectedComplexityMetric} />
              <DistributionChart datasets={datasets} metricKey={selectedComplexityMetric} relative={distributionRelative} />
              <p className="muted small" style={{ marginTop: '.5rem' }}>
                {distributionRelative ? 'Fraction of programs (y)' : 'Count of programs (y)'} at each observed value (x) — top-{datasets[0]?.data?.complexity?.[selectedComplexityMetric]?.top_k?.length ?? 20} most frequent values.
                Mean: {datasets.map(ds => {
                  const d = ds.data.complexity?.[selectedComplexityMetric];
                  return d ? `${ds.label} ${d.mean.toFixed(2)}` : null;
                }).filter(Boolean).join(', ')}.
              </p>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
