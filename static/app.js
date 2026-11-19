(function () {
    const form = document.getElementById('upload-form');
    const input = document.getElementById('image-input');
    const zone = document.getElementById('dropzone');
    const preview = document.getElementById('preview');
    const prompt = zone.querySelector('.drop-prompt');
    const button = document.getElementById('detect-btn');
    const change = document.getElementById('change-btn');

    function showFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        preview.src = URL.createObjectURL(file);
        preview.alt = file.name;
        preview.hidden = false;
        prompt.hidden = true;
        change.hidden = false;
        zone.classList.add('has-image');
    }

    input.addEventListener('change', () => showFile(input.files[0]));

    ['dragenter', 'dragover'].forEach((type) =>
        zone.addEventListener(type, (event) => {
            event.preventDefault();
            zone.classList.add('is-dragging');
        })
    );
    ['dragleave', 'drop'].forEach((type) =>
        zone.addEventListener(type, () => zone.classList.remove('is-dragging'))
    );
    zone.addEventListener('drop', (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (!file) return;
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        showFile(file);
    });

    form.addEventListener('submit', () => {
        button.disabled = true;
        button.textContent = button.dataset.busy;
        zone.classList.add('is-busy');
    });
})();
