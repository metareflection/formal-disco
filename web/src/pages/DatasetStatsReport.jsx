import { useState, useRef, useCallback } from 'react';
import {
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import CodeBlock from '../components/CodeBlock';

const DATASET_COLORS = ['#3949ab', '#e53935', '#2e7d32', '#f57c00', '#7b1fa2', '#00838f'];

// ── Data processing ────────────────────────────────────────────────────────

function buildReport(files) {
  return files.map((f, i) => {
    const methods = f.data.programs.flatMap((p, pi) =>
      p.methods.map(m => ({ ...m, programIndex: pi }))
    );
    return {
      id: f.id,
      label: f.label,
      color: DATASET_COLORS[i % DATASET_COLORS.length],
      data: f.data,
      methods,
      n_programs: f.data.n_programs ?? f.data.programs.length,
    };
  });
}

/**
 * Build distribution data for a metric (assertions or invariants).
 * Returns [{k, ...labels}] where each label maps to the count/pct of methods with that value of k.
 */
function distributionData(datasets, metricKey, cumulative, relative) {
  // Gather all k values across datasets
  const allKs = new Set();
  const perDataset = datasets.map(ds => {
    const counts = {};
    for (const m of ds.methods) {
      const v = m[metricKey];
      counts[v] = (counts[v] || 0) + 1;
      allKs.add(v);
    }
    return counts;
  });

  const sortedKs = [...allKs].sort((a, b) => a - b);

  const rows = sortedKs.map(k => {
    const row = { k };
    datasets.forEach((ds, i) => {
      const counts = perDataset[i];
      let value;
      if (cumulative) {
        // >= k: sum all counts where key >= k
        value = 0;
        for (const [key, cnt] of Object.entries(counts)) {
          if (Number(key) >= k) value += cnt;
        }
      } else {
        value = counts[k] || 0;
      }
      if (relative) {
        value = ds.methods.length > 0 ? value / ds.methods.length : 0;
      }
      row[ds.label] = value;
    });
    return row;
  });

  return rows;
}

/**
 * Find the top N methods by a metric across all datasets.
 * Returns [{datasetLabel, method, program}] sorted descending.
 */
function topMethods(datasets, metricKey, n) {
  const all = [];
  for (const ds of datasets) {
    for (const m of ds.methods) {
      all.push({
        datasetLabel: ds.label,
        datasetColor: ds.color,
        name: m.name,
        value: m[metricKey],
        source: ds.data.programs[m.programIndex].source,
      });
    }
  }
  all.sort((a, b) => b.value - a.value);
  return all.slice(0, n);
}

// ── Toggle button ─────────────────────────────────────────────────────────

function ToggleButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '3px 12px',
        borderRadius: 20,
        border: '2px solid',
        borderColor: active ? '#3949ab' : '#ddd',
        background: active ? '#eef0ff' : '#fff',
        color: active ? '#3949ab' : '#888',
        fontSize: '.78rem',
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all .15s',
      }}
    >
      {children}
    </button>
  );
}

// ── Distribution chart section ────────────────────────────────────────────

function DistributionSection({ datasets, metricKey, title, description }) {
  const [cumulative, setCumulative] = useState(true);
  const [relative, setRelative] = useState(false);

  const chartData = distributionData(datasets, metricKey, cumulative, relative);

  const yLabel = cumulative
    ? (relative ? '% of methods with >= K' : 'Methods with >= K')
    : (relative ? '% of methods with = K' : 'Methods with = K');

  const yFormatter = relative
    ? v => `${(v * 100).toFixed(1)}%`
    : v => v.toLocaleString();

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.75rem' }}>
        <div>
          <h3 style={{ fontSize: '.875rem', fontWeight: 600, color: '#555', margin: 0 }}>{title}</h3>
          {description && <p className="muted small" style={{ margin: '4px 0 0' }}>{description}</p>}
        </div>
        <div style={{ display: 'flex', gap: '.5rem' }}>
          <ToggleButton active={cumulative} onClick={() => setCumulative(c => !c)}>
            {cumulative ? '>= K (cumulative)' : '= K (exact)'}
          </ToggleButton>
          <ToggleButton active={relative} onClick={() => setRelative(r => !r)}>
            {relative ? '%' : 'absolute'}
          </ToggleButton>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ bottom: 8, left: 4, right: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="k"
            tick={{ fontSize: 11 }}
            label={{ value: 'K', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis
            tickFormatter={yFormatter}
            tick={{ fontSize: 11 }}
            width={72}
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 10, fontSize: 10 }}
          />
          <Tooltip formatter={(v, name) => [yFormatter(v), name]} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {datasets.map(ds => (
            <Bar key={ds.label} dataKey={ds.label} fill={ds.color} radius={[3, 3, 0, 0]} maxBarSize={40} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Top methods table ─────────────────────────────────────────────────────

function TopMethodsTable({ datasets, metricKey, metricLabel }) {
  const [expanded, setExpanded] = useState(null);
  const top = topMethods(datasets, metricKey, 10);

  if (top.length === 0) return <p className="muted small">No data.</p>;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="stat-table" style={{ width: '100%' }}>
        <thead>
          <tr>
            <th style={{ width: 40 }}>#</th>
            <th>Method</th>
            <th>Dataset</th>
            <th style={{ width: 80 }}>{metricLabel}</th>
            <th style={{ width: 80 }}></th>
          </tr>
        </thead>
        <tbody>
          {top.map((entry, idx) => (
            <>
              <tr key={idx}>
                <td className="muted">{idx + 1}</td>
                <td style={{ fontFamily: 'monospace', fontSize: '.82rem' }}>{entry.name}</td>
                <td>
                  <span className="color-dot" style={{ background: entry.datasetColor }} />
                  {entry.datasetLabel}
                </td>
                <td style={{ fontWeight: 600 }}>{entry.value}</td>
                <td>
                  <button
                    onClick={() => setExpanded(expanded === idx ? null : idx)}
                    style={{
                      padding: '2px 10px',
                      borderRadius: 4,
                      border: '1px solid #ccc',
                      background: expanded === idx ? '#eef0ff' : '#fff',
                      fontSize: '.75rem',
                      cursor: 'pointer',
                    }}
                  >
                    {expanded === idx ? 'Hide' : 'Show'}
                  </button>
                </td>
              </tr>
              {expanded === idx && (
                <tr key={`${idx}-code`}>
                  <td colSpan={5} style={{ padding: 0 }}>
                    <CodeBlock code={entry.source} />
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function DatasetStatsReport() {
  const [files, setFiles] = useState([]);
  const [datasets, setDatasets] = useState(null);
  const fileInputRef = useRef(null);

  const addFiles = useCallback(async (fileList) => {
    const incoming = [];
    for (const file of Array.from(fileList)) {
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (!data.programs || !Array.isArray(data.programs)) {
          alert(`${file.name} doesn't look like a dataset stats file (missing programs array).`);
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

  return (
    <div className="page">
      <h1>Dafny Dataset Statistics</h1>

      {/* ── File loading ── */}
      <section className="section">
        <h2>Load Dataset Files</h2>
        <p className="muted small" style={{ marginBottom: '1rem' }}>
          Drop one or more <code>.programs.json</code> files produced by{' '}
          <code>materialize.py export-programs</code> or <code>materialize.py export-dafnybench</code>.
          Multiple files are compared side by side.
        </p>
        <div
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
        >
          Drag &amp; drop dataset JSON files here, or click to browse
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
              <tr><th>Label</th><th>Programs</th><th>Methods</th><th></th></tr>
            </thead>
            <tbody>
              {files.map(f => {
                const nMethods = f.data.programs.reduce((s, p) => s + p.methods.length, 0);
                return (
                  <tr key={f.id}>
                    <td>
                      <input
                        className="label-input"
                        value={f.label}
                        onChange={e => updateLabel(f.id, e.target.value)}
                      />
                    </td>
                    <td className="muted">{(f.data.n_programs ?? f.data.programs.length).toLocaleString()}</td>
                    <td className="muted">{nMethods.toLocaleString()}</td>
                    <td><button className="btn-remove" onClick={() => removeFile(f.id)}>x</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <button className="btn-primary" onClick={handleBuild} disabled={!files.length}>
          Build Report
        </button>
      </section>

      {datasets && (
        <>
          {/* ── Summary ── */}
          <section className="section">
            <h2>Summary</h2>
            <table className="stat-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Programs</th>
                  <th>Methods</th>
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
                    <td>{ds.methods.length.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {/* ── Assertions distribution ── */}
          <section className="section">
            <DistributionSection
              datasets={datasets}
              metricKey="assertions"
              title="Assertions per Method"
              description="Distribution of the number of assert statements per method/function/lemma."
            />
          </section>

          {/* ── Invariants distribution ── */}
          <section className="section">
            <DistributionSection
              datasets={datasets}
              metricKey="invariants"
              title="Loop Invariants per Method"
              description="Distribution of the number of loop invariant annotations per method."
            />
          </section>

          {/* ── Top 10 by assertions ── */}
          <section className="section">
            <h2>Top 10 Methods by Assertions</h2>
            <p className="muted small" style={{ marginBottom: '1rem' }}>
              Methods with the most assert statements. Click "Show" to see the full program.
            </p>
            <TopMethodsTable datasets={datasets} metricKey="assertions" metricLabel="Asserts" />
          </section>

          {/* ── Top 10 by invariants ── */}
          <section className="section">
            <h2>Top 10 Methods by Loop Invariants</h2>
            <p className="muted small" style={{ marginBottom: '1rem' }}>
              Methods with the most loop invariant annotations. Click "Show" to see the full program.
            </p>
            <TopMethodsTable datasets={datasets} metricKey="invariants" metricLabel="Invariants" />
          </section>
        </>
      )}
    </div>
  );
}
