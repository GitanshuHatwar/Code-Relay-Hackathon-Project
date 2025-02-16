// Initialize data
let uploads = [
    { id: 1, name: 'Lecture_1.pdf', type: 'file', date: '2025-01-28', url: '#' },
    { id: 2, name: 'Assignment_Guide.docx', type: 'file', date: '2025-01-27', url: '#' }
];

let quizzes = [
    { id: 1, name: 'Math Quiz', type: 'quiz', date: new Date().toISOString(), url: 'https://example.com/math-quiz' },
    { id: 2, name: 'Science Quiz', type: 'quiz', date: new Date().toISOString(), url: 'https://example.com/science-quiz' }
];

// Initialize dashboard
function initDashboard() {
    updateStats();
    renderFiles();
    renderQuizzes();
}

// Update statistics
function updateStats() {
    document.getElementById('totalFiles').textContent = uploads.length;
    document.getElementById('totalQuizzes').textContent = quizzes.length;
}

// Render files
function renderFiles() {
    const availableFilesContainer = document.getElementById('availableFiles');
    availableFilesContainer.innerHTML = uploads.map(item => `
        <div class="file-item">
            <h4>${item.name}</h4>
            <p>Type: ${item.type}</p>
            <p>Date: ${item.date}</p>
            ${item.url ? `<a href="${item.url}" class="btn" target="_blank">Download</a>` : ''}
        </div>
    `).join('');
}

// Render quizzes
function renderQuizzes() {
    const availableQuizzesContainer = document.getElementById('availableQuizzes');
    availableQuizzesContainer.innerHTML = quizzes.map(item => {
        const expiryTime = new Date(new Date(item.date).getTime() + 24 * 60 * 60 * 1000); // 24 hours from creation
        const timeLeft = calculateTimeLeft(expiryTime);

        return `
            <div class="quiz-item">
                <h4>${item.name}</h4>
                <p>Type: ${item.type}</p>
                <p>Date: ${new Date(item.date).toLocaleString()}</p>
                <p class="expiry">Expires in: ${timeLeft}</p>
                ${item.url ? `<a href="${item.url}" class="btn" target="_blank">Start Quiz</a>` : ''}
            </div>
        `;
    }).join('');
}

// Calculate time left for quiz expiry
function calculateTimeLeft(expiryTime) {
    const now = new Date();
    const timeDiff = expiryTime - now;

    if (timeDiff <= 0) {
        return 'Expired';
    }

    const hours = Math.floor((timeDiff / (1000 * 60 * 60)) % 24);
    const minutes = Math.floor((timeDiff / (1000 * 60)) % 60);
    const seconds = Math.floor((timeDiff / 1000) % 60);

    return `${hours}h ${minutes}m ${seconds}s`;
}

// Search functionality for files
function searchFiles() {
    const searchTerm = event.target.value.toLowerCase();
    const container = document.getElementById('availableFiles');
    
    container.querySelectorAll('.file-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        const matches = text.includes(searchTerm);
        item.classList.toggle('highlight', matches && searchTerm !== '');
        item.style.display = matches || searchTerm === '' ? 'block' : 'none';
    });
}

// Search functionality for quizzes
function searchQuizzes() {
    const searchTerm = event.target.value.toLowerCase();
    const container = document.getElementById('availableQuizzes');
    
    container.querySelectorAll('.quiz-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        const matches = text.includes(searchTerm);
        item.classList.toggle('highlight', matches && searchTerm !== '');
        item.style.display = matches || searchTerm === '' ? 'block' : 'none';
    });
}

// Update quiz timers every second
setInterval(() => {
    renderQuizzes();
}, 1000);

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', initDashboard);