import { useState, useRef, useCallback, Fragment } from 'react';
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import CodeBlock from '../components/CodeBlock';

function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible">
      <button className="collapsible-header" onClick={() => setOpen(o => !o)}>
        {open ? '▼' : '▶'} {title}
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </div>
  );
}

const MODEL_COLORS = ['#3949ab', '#e53935', '#2e7d32', '#f57c00', '#7b1fa2', '#00838f'];

// ── Data processing ──────────────────────────────────────────────────────────

function buildReport(files) {
  const models = files.map((f, i) => ({
    label: f.label,
    color: MODEL_COLORS[i % MODEL_COLORS.length],
    byName: new Map(
      (f.data.results || []).map(r => [r.example_name, r])
    ),
    total: (f.data.results || []).length,
    successCount: (f.data.results || []).filter(r => r.success).length,
  }));

  const seen = new Set();
  const examples = [];
  for (const f of files) {
    for (const r of f.data.results || []) {
      if (r.example_name && !seen.has(r.example_name)) {
        seen.add(r.example_name);
        examples.push(r.example_name);
      }
    }
  }

  const maxAttempts = Math.max(
    1,
    ...files.flatMap(f => (f.data.results || []).map(r => r.num_attempts || 0))
  );
  const cumulativeData = [];
  for (let k = 1; k <= maxAttempts; k++) {
    const point = { attempt: k };
    for (const m of models) {
      const fixed = [...m.byName.values()]
        .filter(r => r.success && (r.num_attempts || 0) <= k).length;
      point[m.label] = m.total > 0 ? +((fixed / m.total) * 100).toFixed(1) : 0;
    }
    cumulativeData.push(point);
  }

  const uniqueFixes = models.map(m => {
    const others = models.filter(o => o !== m);
    const count = examples.filter(name => {
      const r = m.byName.get(name);
      return r?.success && others.every(o => !o.byName.get(name)?.success);
    }).length;
    return { name: m.label, value: count, color: m.color };
  });

  return { models, examples, cumulativeData, uniqueFixes };
}

// ── Sub-components ────────────────────────────────────────────────────────────

function LemmaResultDetail({ result, language, languageLabel }) {
  if (!result) {
    return <div className="no-data">No data for this model on this example.</div>;
  }
  const {
    success, num_attempts = 0, verification_outcome,
    generated_body, program, lemma_name,
    initial_outcome, initial_notes,
    final_program, final_notes,
  } = result;
  return (
    <div className="interaction-log">
      <div className={`result-banner ${success ? 'success' : 'failure'}`}>
        {success
          ? `✓ Synthesized in ${num_attempts} attempt${num_attempts !== 1 ? 's' : ''}`
          : `✗ Failed after ${num_attempts} attempt${num_attempts !== 1 ? 's' : ''}`}
        {verification_outcome && verification_outcome !== 'SUCCESS'
          ? ` — ${verification_outcome}` : ''}
      </div>

      {program && (
        <div className="attempt-step">
          <div className="attempt-header">
            Program (before)
            {lemma_name && (
              <span style={{ fontWeight: 400, marginLeft: '0.75rem', color: '#555', textTransform: 'none', letterSpacing: 0 }}>
                — synthesizing body of <code style={{ background: '#eef0ff', padding: '1px 6px', borderRadius: 4 }}>{lemma_name}</code>
              </span>
            )}
          </div>
          <CodeBlock code={program} language={language} />

          {initial_outcome && (
            <div style={{ marginTop: '0.5rem' }}>
              <strong>Initial verification:</strong>{' '}
              <span className={`tab-status ${initial_outcome === 'SUCCESS' ? 'ok' : 'fail'}`}>
                {initial_outcome}
              </span>
            </div>
          )}

          {initial_notes && (
            <Collapsible title={`${languageLabel} output (before)`} defaultOpen>
              <pre className="dafny-output">{initial_notes}</pre>
            </Collapsible>
          )}
        </div>
      )}

      <div className="attempt-step">
        <div className="attempt-header">Generated body</div>
        {generated_body
          ? <CodeBlock code={generated_body} language={language} />
          : <p className="muted small" style={{ padding: '0.5rem' }}>No body was generated.</p>
        }
      </div>

      {final_program && (
        <div className={`attempt-step${success ? ' success-step' : ''}`}>
          <div className="attempt-header">Program (after)</div>
          <CodeBlock code={final_program} language={language} />

          {final_notes && (
            <Collapsible title={`${languageLabel} output (after)`} defaultOpen>
              <pre className="dafny-output">{final_notes}</pre>
            </Collapsible>
          )}
        </div>
      )}
    </div>
  );
}

function ExampleDetail({ name, models, language, languageLabel }) {
  const [activeTab, setActiveTab] = useState(0);
  return (
    <div className="problem-detail">
      <div className="tabs">
        {models.map((m, i) => {
          const r = m.byName.get(name);
          return (
            <button
              key={m.label}
              className={`tab${activeTab === i ? ' active' : ''}`}
              style={{ '--tab-color': m.color }}
              onClick={() => setActiveTab(i)}
            >
              <span className="color-dot" style={{ background: m.color }} />
              {m.label}
              {r && (
                <span className={`tab-status ${r.success ? 'ok' : 'fail'}`}>
                  {r.success ? '✓' : '✗'}
                </span>
              )}
            </button>
          );
        })}
      </div>
      <div className="tab-content">
        <LemmaResultDetail
          result={models[activeTab]?.byName.get(name)}
          language={language}
          languageLabel={languageLabel}
        />
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function LemmaReport({
  title = 'Lemma Synthesis Evaluation Report',
  language = 'dafny',
  languageLabel = 'Dafny',
} = {}) {
  const [files, setFiles] = useState([]);
  const [report, setReport] = useState(null);
  const [search, setSearch] = useState('');
  const [expandedExample, setExpandedExample] = useState(null);
  const fileInputRef = useRef(null);

  const addFiles = useCallback(async (fileList) => {
    const incoming = [];
    for (const file of Array.from(fileList)) {
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        incoming.push({
          id: Math.random().toString(36).slice(2),
          label: file.name.replace(/\.json$/, ''),
          data,
        });
      } catch (e) {
        alert(`Failed to parse ${file.name}: ${e.message}`);
      }
    }
    setFiles(prev => [...prev, ...incoming]);
    setReport(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const removeFile = (id) => { setFiles(prev => prev.filter(f => f.id !== id)); setReport(null); };
  const updateLabel = (id, label) => { setFiles(prev => prev.map(f => f.id === id ? { ...f, label } : f)); setReport(null); };

  const handleBuild = () => {
    if (!files.length) return;
    setReport(buildReport(files));
    setExpandedExample(null);
    setSearch('');
  };

  const filteredExamples = report
    ? report.examples.filter(e => e.toLowerCase().includes(search.toLowerCase()))
    : [];

  return (
    <div className="page">
      <h1>{title}</h1>

      {/* ── File loading ── */}
      <section className="section">
        <h2>Load Result Files</h2>
        <div
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
        >
          Drag &amp; drop JSON result files here, or click to browse
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
              <tr><th>Label</th><th>Examples</th><th>Success</th><th></th></tr>
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
                  <td className="muted">{f.data.results?.length ?? '?'}</td>
                  <td className="muted">
                    {f.data.success_count ?? '?'} / {f.data.num_examples ?? f.data.results?.length ?? '?'}
                    {f.data.success_rate != null
                      ? ` (${(f.data.success_rate * 100).toFixed(1)}%)`
                      : ''}
                  </td>
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

      {report && (
        <>
          {/* ── Statistics ── */}
          <section className="section">
            <h2>Statistics</h2>
            <div className="stats-grid">

              <div className="stat-card">
                <h3>Overall Success Rate</h3>
                <table className="stat-table">
                  <thead>
                    <tr><th>Model</th><th>Solved</th><th>Total</th><th>Rate</th></tr>
                  </thead>
                  <tbody>
                    {report.models.map(m => (
                      <tr key={m.label}>
                        <td>
                          <span className="color-dot" style={{ background: m.color }} />
                          {m.label}
                        </td>
                        <td>{m.successCount}</td>
                        <td>{m.total}</td>
                        <td>
                          <strong>
                            {m.total > 0 ? ((m.successCount / m.total) * 100).toFixed(1) : 0}%
                          </strong>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="stat-card">
                <h3>Cumulative Success by Attempts</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={report.cumulativeData} margin={{ bottom: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="attempt"
                      label={{ value: 'Attempts allowed', position: 'insideBottom', offset: -8, fontSize: 11 }}
                    />
                    <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} width={45} />
                    <Tooltip formatter={(v, name) => [`${v}%`, name]} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {report.models.map(m => (
                      <Line
                        key={m.label}
                        type="monotone"
                        dataKey={m.label}
                        stroke={m.color}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        activeDot={{ r: 5 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="stat-card">
                <h3>Unique Solves</h3>
                <p className="muted small" style={{ marginBottom: '.75rem' }}>
                  Examples solved exclusively by each model.
                </p>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={report.uniqueFixes} margin={{ bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} width={32} />
                    <Tooltip />
                    <Bar dataKey="value" name="Unique solves" radius={[4, 4, 0, 0]}>
                      {report.uniqueFixes.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

            </div>
          </section>

          {/* ── Example-by-example ── */}
          <section className="section">
            <h2>Example Details</h2>
            <div className="search-bar">
              <input
                type="search"
                placeholder="Filter by example name…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
              <span className="muted small">{filteredExamples.length} / {report.examples.length} examples</span>
            </div>

            <table className="problem-table">
              <thead>
                <tr>
                  <th>Example</th>
                  {report.models.map(m => (
                    <th key={m.label}>
                      <span className="color-dot" style={{ background: m.color }} />
                      {m.label}
                    </th>
                  ))}
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filteredExamples.map(name => {
                  const isExpanded = expandedExample === name;
                  return (
                    <Fragment key={name}>
                      <tr
                        className={`problem-row${isExpanded ? ' expanded' : ''}`}
                        onClick={() => setExpandedExample(isExpanded ? null : name)}
                      >
                        <td className="program-name" title={name}>{name}</td>
                        {report.models.map(m => {
                          const r = m.byName.get(name);
                          if (!r) return <td key={m.label} className="status-cell missing">—</td>;
                          return (
                            <td key={m.label} className={`status-cell ${r.success ? 'ok' : 'fail'}`}>
                              {r.success ? `✓ (${r.num_attempts})` : '✗'}
                            </td>
                          );
                        })}
                        <td className="expand-cell">{isExpanded ? '▲' : '▼'}</td>
                      </tr>
                      {isExpanded && (
                        <tr>
                          <td colSpan={report.models.length + 2}>
                            <ExampleDetail
                              name={name}
                              models={report.models}
                              language={language}
                              languageLabel={languageLabel}
                            />
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
