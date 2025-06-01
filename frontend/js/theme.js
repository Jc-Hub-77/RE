// frontend/js/theme.js
console.log("theme.js loaded");

document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeSwitch');
    const currentTheme = localStorage.getItem('theme');

    function applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-mode');
            if (themeToggle) themeToggle.checked = true;
        } else {
            document.body.classList.remove('dark-mode');
            if (themeToggle) themeToggle.checked = false;
        }
    }

    // Apply saved theme on initial load
    if (currentTheme) {
        applyTheme(currentTheme);
    } else {
        // Default to light theme or check system preference if desired
        // const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        // if (prefersDark) applyTheme('dark'); else applyTheme('light');
        applyTheme('light'); // Default to light
    }

    if (themeToggle) {
        themeToggle.addEventListener('change', function() {
            if (this.checked) {
                localStorage.setItem('theme', 'dark');
                applyTheme('dark');
            } else {
                localStorage.setItem('theme', 'light');
                applyTheme('light');
            }
        });
    } else {
        console.warn("Theme toggle switch with ID 'themeSwitch' not found.");
    }
});
