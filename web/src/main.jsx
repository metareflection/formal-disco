import React from 'react';
import ReactDOM from 'react-dom/client';
import hljs from 'highlight.js/lib/core';
import rust from 'highlight.js/lib/languages/rust';
import App from './App';
import './dafny.min.js';
import './index.css';

hljs.registerLanguage('rust', rust);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
