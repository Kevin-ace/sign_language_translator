document.addEventListener('DOMContentLoaded', function() {
    const translatedText = document.getElementById('translatedText');
    const historyContent = document.getElementById('historyContent');
    const status = document.getElementById('status');
    
    // Simulated translation updates (to be replaced with actual backend communication)
    function updateTranslation(text) {
        translatedText.textContent = text;
        addToHistory(text);
    }
    
    function addToHistory(text) {
        const timestamp = new Date().toLocaleTimeString();
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.textContent = `[${timestamp}] ${text}`;
        
        historyContent.insertBefore(historyItem, historyContent.firstChild);
        
        // Keep only last 10 items
        while (historyContent.children.length > 10) {
            historyContent.removeChild(historyContent.lastChild);
        }
    }
    
    // Check connection status
    function updateConnectionStatus() {
        const videoFeed = document.getElementById('videoFeed');
        if (videoFeed.complete && videoFeed.naturalHeight !== 0) {
            status.textContent = 'Connected';
            status.style.color = '#27ae60';
        } else {
            status.textContent = 'Disconnected';
            status.style.color = '#c0392b';
        }
    }
    
    // Update connection status every 5 seconds
    setInterval(updateConnectionStatus, 5000);
    
    // Initial connection check
    updateConnectionStatus();
});
