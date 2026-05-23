async function analyze() {
  const input = document.getElementById("messageInput");
  const text = input.value.trim();
  const resultEl = document.getElementById("result");
  const errorEl = document.getElementById("error");
  const btn = document.getElementById("analyzeBtn");
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");

  // Reset
  resultEl.className = "result hidden";
  errorEl.className = "error hidden";

  if (!text) {
    errorEl.textContent = "Please enter a message to analyze.";
    errorEl.className = "error";
    return;
  }

  // Loading state
  btn.disabled = true;
  btnText.classList.add("hidden");
  btnLoader.classList.remove("hidden");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error("Server error: " + res.status);

    const data = await res.json();
    const isSpam = data.prediction === "SPAM";

    document.getElementById("resultBadge").textContent = isSpam ? "🚨" : "✅";
    document.getElementById("resultLabel").textContent = isSpam ? "SPAM Detected" : "Looks Legitimate";
    document.getElementById("confidenceFill").style.width = data.confidence + "%";
    document.getElementById("confidenceText").textContent = data.confidence + "% confidence";

    resultEl.className = "result " + (isSpam ? "spam" : "ham");

  } catch (err) {
    errorEl.textContent = "Something went wrong. Please try again.";
    errorEl.className = "error";
  } finally {
    btn.disabled = false;
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
}

// Allow Enter key in textarea with Ctrl/Cmd
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("messageInput").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
});