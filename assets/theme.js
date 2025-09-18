(() => {
  const LS_KEY = 'site-theme';
  function apply(theme){
    if(theme === 'light') document.body.setAttribute('data-theme','light');
    else document.body.removeAttribute('data-theme');
  }
  function current(){ return document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dim'; }
  function toggle(){
    const next = current() === 'light' ? 'dim' : 'light';
    apply(next);
    try { localStorage.setItem(LS_KEY, next); } catch {}
    const btn = document.getElementById('theme-toggle');
    if(btn) btn.textContent = next === 'light' ? 'Dim Theme' : 'Light Theme';
  }
  // Init
  try {
    const saved = localStorage.getItem(LS_KEY);
    if(saved) apply(saved);
  } catch {}
  window.toggleTheme = toggle;
  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('theme-toggle');
    if(btn) btn.addEventListener('click', toggle);
    if(btn) btn.textContent = current() === 'light' ? 'Dim Theme' : 'Light Theme';
  });
})();

