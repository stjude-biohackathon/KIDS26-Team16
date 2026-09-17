function applyTheme(theme) {
    let effective = theme;
    if (theme === 'auto') {
        effective = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-bs-theme', effective);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById('theme-btn-' + theme);
    if (btn) btn.classList.add('active');
}

function setTheme(theme) {
    localStorage.setItem('scogs-theme', theme);
    applyTheme(theme);
}

document.addEventListener('DOMContentLoaded', function() {
    const saved = localStorage.getItem('scogs-theme') || 'auto';
    applyTheme(saved);
});

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
    if ((localStorage.getItem('scogs-theme') || 'auto') === 'auto') {
        applyTheme('auto');
    }
});
