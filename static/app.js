function predictExpression() {
    const file = document.getElementById("imageInput").files[0];
    if (!file) {
        alert("Upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    document.getElementById("loading").classList.remove("hidden");

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("loading").classList.add("hidden");

        document.getElementById("inputImage").src = data.image_url;
        document.getElementById("inputImage").classList.remove("hidden");

        const predictionText = document.getElementById("predictionText");
        predictionText.innerText = `Detected: ${data.prediction}`;
        predictionText.classList.remove("hidden");

        const confidenceText = document.getElementById("confidenceText");
        confidenceText.innerText = `Confidence: ${data.confidence}%`;
        confidenceText.classList.remove("hidden");
    })
    .catch(err => {
        alert("Something went wrong!");
        console.error("Fetch error:", err);
    });
}

