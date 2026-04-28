import { useState } from 'react';
import CodeBlock from './CodeBlock';
import DiffBlock from './DiffBlock';

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

function AttemptStep({ step, index, isLast, overallSuccess, language, languageLabel }) {
  const isErrorResult = typeof step.result === 'string' && step.result.startsWith('Error:');
  const hasProgram = typeof step.result === 'string' && !isErrorResult;

  return (
    <div className={`attempt-step${isLast && overallSuccess ? ' success-step' : ''}`}>
      <div className="attempt-header">Attempt {index + 1}</div>

      <Collapsible title="Program (before)" defaultOpen={index === 0}>
        <CodeBlock code={step.program || ''} language={language} />
      </Collapsible>

      {step.notes && (
        <Collapsible title={`${languageLabel} output`}>
          <pre className="dafny-output">{step.notes}</pre>
        </Collapsible>
      )}

      {step.diff && (
        <Collapsible title="LLM diff" defaultOpen>
          <DiffBlock diff={step.diff} />
        </Collapsible>
      )}

      {isErrorResult && (
        <div className="error-box" style={{ marginBottom: '.5rem' }}>{step.result}</div>
      )}

      {hasProgram && (
        <Collapsible title="After applying diff" defaultOpen>
          <CodeBlock code={step.result} language={language} />
        </Collapsible>
      )}

      {step.result_notes && (
        <Collapsible title={`${languageLabel} output after`} defaultOpen={isLast}>
          <pre className="dafny-output">{step.result_notes}</pre>
        </Collapsible>
      )}

      {step.diff_vs_ground_truth && (
        <Collapsible title="Diff vs ground truth">
          <DiffBlock diff={step.diff_vs_ground_truth} />
        </Collapsible>
      )}

      {step.ground_truth_program && (
        <Collapsible title="Ground truth program">
          <CodeBlock code={step.ground_truth_program} language={language} />
        </Collapsible>
      )}
    </div>
  );
}

export default function InteractionLog({ result, language = 'dafny', languageLabel = 'Dafny' }) {
  if (!result) {
    return <div className="no-data">No data for this model on this problem.</div>;
  }

  const log = result.interaction_log || [];
  const { success, num_attempts = 0, verification_outcome } = result;

  return (
    <div className="interaction-log">
      <div className={`result-banner ${success ? 'success' : 'failure'}`}>
        {success
          ? `✓ Fixed in ${num_attempts} attempt${num_attempts !== 1 ? 's' : ''}`
          : `✗ Failed after ${num_attempts} attempt${num_attempts !== 1 ? 's' : ''}`}
        {verification_outcome && verification_outcome !== 'SUCCESS'
          ? ` — ${verification_outcome}` : ''}
      </div>

      {log.length === 0 && (
        <p className="muted small">No interaction log recorded.</p>
      )}

      {log.map((step, i) => (
        <AttemptStep
          key={i}
          step={step}
          index={i}
          isLast={i === log.length - 1}
          overallSuccess={success}
          language={language}
          languageLabel={languageLabel}
        />
      ))}
    </div>
  );
}
