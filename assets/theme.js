// Apply the saved preference before the page paints. Storage may be disabled.
try {
  const preference = localStorage.getItem('theme');
  if (preference === 'light' || preference === 'dark') document.documentElement.dataset.theme = preference;
} catch (_) { /* System preference remains the fallback. */ }
