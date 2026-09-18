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

// Instantaneous visual feedback on outcome pill selection
document.addEventListener('click', function(e) {
    const btn = e.target.closest('.outcome-summary-btn');
    if (btn) {
        document.querySelectorAll('.outcome-summary-btn').forEach(b => b.classList.remove('outcome-pill-active'));
        btn.classList.add('outcome-pill-active');
    }
});

// Synchronize sidebar dropdown changes to overview outcome pill highlight
document.addEventListener('change', function(e) {
    if (e.target && e.target.id === 'selected_outcome_num') {
        const val = String(e.target.value);
        document.querySelectorAll('.outcome-summary-btn').forEach(b => {
            if (b.getAttribute('data-outcome-num') === val) {
                b.classList.add('outcome-pill-active');
            } else {
                b.classList.remove('outcome-pill-active');
            }
        });
    }
});

// Synchronize active class on segmented radio controls
function updateSegmentedRadios() {
    document.querySelectorAll('.sidebar-mode-switcher, .sidebar-segmented-radios').forEach(function(container) {
        container.querySelectorAll('.radio, .form-check').forEach(function(el) {
            const input = el.querySelector('input[type="radio"]');
            if (input && input.checked) {
                el.classList.add('is-active');
            } else {
                el.classList.remove('is-active');
            }
        });
    });
}

document.addEventListener('change', function(e) {
    if (e.target && e.target.type === 'radio') {
        updateSegmentedRadios();
    }
});

document.addEventListener('DOMContentLoaded', updateSegmentedRadios);
let _segmentedTimer = null;
const debouncedUpdateSegmented = function() {
    if (_segmentedTimer) clearTimeout(_segmentedTimer);
    _segmentedTimer = setTimeout(updateSegmentedRadios, 50);
};
const observer = new MutationObserver(debouncedUpdateSegmented);
if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
} else {
    document.addEventListener('DOMContentLoaded', function() {
        observer.observe(document.body, { childList: true, subtree: true });
    });
}
