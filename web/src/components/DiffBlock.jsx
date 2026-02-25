export default function DiffBlock({ diff }) {
  const lines = (diff || '').split('\n');
  return (
    <pre className="diff-block">
      {lines.map((line, i) => {
        let cls = 'diff-line';
        if (line.startsWith('+')) cls += ' diff-add';
        else if (line.startsWith('-')) cls += ' diff-del';
        else if (line.startsWith('@@')) cls += ' diff-hunk';
        return <span key={i} className={cls}>{line}{'\n'}</span>;
      })}
    </pre>
  );
}
