const themeButton = document.querySelector('.theme-toggle');
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
function isDark() {
  return document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark'
    : systemTheme.matches;
}
function updateThemeLabel() {
  themeButton.setAttribute('aria-label', `Switch to ${isDark() ? 'light' : 'dark'} theme`);
}
if (themeButton) {
  themeButton.hidden = false;
  updateThemeLabel();
  themeButton.addEventListener('click', () => {
    const theme = isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('theme', theme); } catch (_) { /* The toggle still works. */ }
    updateThemeLabel();
  });
  systemTheme.addEventListener('change', updateThemeLabel);
}
// Ordinary anchor links continue to work without JavaScript.
if ('IntersectionObserver' in window) {
  const links = [...document.querySelectorAll('nav a')];
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) {
        if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
    }
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
}
