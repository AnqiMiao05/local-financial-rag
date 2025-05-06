// rag_web/app.js

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("queryInput");
  const button = document.getElementById("askButton");
  const result = document.getElementById("result");
  const historyList = document.getElementById("historyList");
  const clearHistoryButton = document.getElementById("clearHistory");

  const updateHistory = (query) => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.textContent = query;
    item.onclick = () => {
      input.value = query;
      button.click();
    };
    historyList.prepend(item);
  };

  const clearHistory = () => {
    historyList.innerHTML = "";
  };

  clearHistoryButton.onclick = clearHistory;

  button.onclick = async () => {
    const query = input.value.trim();
    if (!query) return;

    result.innerHTML = "Thinking...";
    result.style.display = "block";
    updateHistory(query);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });

      const data = await res.json();
      const references = (data.references || [])
        .map(
          (ref) => `<li>${ref.title} <small>(ID: ${ref.id})</small></li>`
        )
        .join("");

      result.innerHTML = `<p>${data.answer}</p><ul>${references}</ul>`;
    } catch (err) {
      result.innerHTML = "❌ Failed to fetch response.";
      console.error("Request error:", err);
    }
  };
});
