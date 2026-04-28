import { useEffect, useRef } from 'react';
import hljs from 'highlight.js/lib/core';
import 'highlight.js/styles/atom-one-dark.css';

// Dafny isn't in highlight.js's built-in set; C# gives reasonable highlighting
// since they share keywords (method, function, returns, requires, ensures, etc.)
// hljs.registerLanguage('csharp', csharp);

export default function CodeBlock({ code, language = 'dafny' }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      // Remove the flag so hljs re-highlights after updates
      ref.current.removeAttribute('data-highlighted');
      hljs.highlightElement(ref.current);
    }
  }, [code, language]);

  return (
    <div className="code-block-wrap">
      <pre><code ref={ref} className={`language-${language}`}>{code}</code></pre>
    </div>
  );
}
