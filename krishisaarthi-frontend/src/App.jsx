import { useState } from "react";
import axios from "axios";
import "./App.css";

const knowledgeSections = [
  { key: "symptoms", title: "Symptoms", subtitle: "What you may observe", icon: "S" },
  { key: "favorable_conditions", title: "Favorable conditions", subtitle: "What can encourage it", icon: "C" },
  { key: "precautions", title: "Precautions", subtitle: "Steps to take now", icon: "P" },
  { key: "management", title: "Management", subtitle: "Recommended care", icon: "M" },
  { key: "prevention", title: "Prevention", subtitle: "Reduce recurrence", icon: "R" },
];

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const detectDisease = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("file", image);
    try {
      setLoading(true);
      setError("");
      const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (requestError) {
      console.error(requestError);
      setError("We couldn't connect to the analysis service. Check that the backend is running and try again.");
    } finally {
      setLoading(false);
    }
  };

  const knowledge = result?.knowledge ?? {};

  return (
    <div className="app">
      <header className="navbar">
        <a className="logo" href="#top" aria-label="KrishiSaarthi AI home">
          <span className="logo-mark" aria-hidden="true">KS</span> KrishiSaarthi <span>AI</span>
        </a>
        <div className="nav-text">Explainable crop health, made simple</div>
      </header>

      <main className="container" id="top">
        <section className="intro">
          <div className="eyebrow">CROP HEALTH ASSISTANT</div>
          <h1>A clearer picture of your plant's health.</h1>
          <p>Upload a leaf photo to identify a possible disease and see what the AI noticed.</p>
        </section>

        <section className="workspace" aria-label="Crop disease analysis">
          <div className="upload-card">
            <div className="section-heading">
              <span className="step">01</span>
              <div><h2>Add a leaf photo</h2><p>Use a clear, well-lit image for best results.</p></div>
            </div>
            <label className={`upload-area${preview ? " has-preview" : ""}`}>
              {preview ? <img src={preview} alt="Uploaded leaf" className="preview" /> : <>
                <span className="upload-icon" aria-hidden="true">+</span>
                <strong>Choose an image</strong><span className="upload-hint">or drag it here</span>
                <span className="file-types">JPG, JPEG or PNG</span>
              </>}
              <input type="file" accept="image/jpeg,image/png,image/jpg" onChange={handleImageChange} />
            </label>
            {image && <div className="file-name">{image.name}</div>}
            <button onClick={detectDisease} disabled={loading || !image} className="detect-button">
              {loading ? <><span className="spinner" /> Analyzing leaf...</> : <>Analyze leaf <span aria-hidden="true">-&gt;</span></>}
            </button>
            <p className="privacy-note">Your image is used only to create this prediction.</p>
          </div>

          <section className={`analysis-panel${result ? " has-result" : ""}`} aria-live="polite">
            {!result ? <div className="empty-state">
              <div className="empty-illustration" aria-hidden="true"><span>+</span></div>
              <div className="eyebrow">YOUR ANALYSIS</div>
              <h2>{loading ? "Looking closely at your leaf..." : "Your results will appear here"}</h2>
              <p>{loading ? "The model is checking the image for visual signs of disease." : "Add a leaf photo, then select Analyze leaf to see the predicted disease and its explanation."}</p>
              {error && <p className="error-message" role="alert">{error}</p>}
              <div className="result-points"><span><i>1</i> Disease prediction</span><span><i>2</i> Visual explanation</span><span><i>3</i> Care guidance</span></div>
            </div> : <div className="results">
              <section className="analysis-core" aria-label="Disease prediction and visual explanation">
                <div className="eyebrow">01 / PREDICTION</div>
                <div className="result-title-row">
                  <div>
                    <span className="result-label">PREDICTED DISEASE</span>
                    <h2>{knowledge.disease || result.disease}</h2>
                    {knowledge.crop && <p className="crop-name">Crop: {knowledge.crop}</p>}
                  </div>
                  <div className="confidence-badge"><strong>{(result.confidence * 100).toFixed(1)}%</strong><span>confidence</span></div>
                </div>

                <div className="explanation-card">
                  <div className="explanation-copy">
                    <span className="result-label">VISUAL EXPLANATION</span>
                    <h3>What the AI focused on</h3>
                    <p>Highlighted areas show the parts of the leaf that contributed most to this prediction.</p>
                  </div>
                  {result.gradcam && <img src={`data:image/png;base64,${result.gradcam}`} alt="AI visual explanation highlighting relevant leaf regions" className="gradcam" />}
                </div>

                {knowledge.cause && <div className="cause-card">
                  <span className="cause-mark" aria-hidden="true">i</span>
                  <div><span className="result-label">ABOUT THIS DISEASE</span><p>{knowledge.cause}</p></div>
                </div>}
                {(result.reason || result.advice) && <div className="ai-notes">
                  <span className="result-label">AI NOTES</span>
                  {result.reason && <p>{result.reason}</p>}
                  {result.advice && <p>{result.advice}</p>}
                </div>}
                <p className="disclaimer">AI predictions are a helpful starting point. Confirm with a local agricultural expert before taking action.</p>
              </section>

              <section className="guidance-panel" aria-label="Disease guidance">
                <div className="guidance-heading">
                  <div><span className="result-label">02 / FIELD GUIDE</span><h3>Understand and care for your crop</h3></div>
                  <span className="guide-count">{knowledgeSections.filter(({ key }) => knowledge[key]?.length).length} topics</span>
                </div>
                {knowledgeSections.map(({ key, title, subtitle, icon }, index) => {
                  const items = knowledge[key];
                  if (!items?.length) return null;
                  return <article className="knowledge-section" key={key}>
                    <div className="knowledge-heading">
                      <span className="knowledge-icon" aria-hidden="true">{icon}</span>
                      <div className="knowledge-title"><strong>{title}</strong><small>{subtitle}</small></div>
                      <span className="knowledge-index">{String(index + 1).padStart(2, "0")}</span>
                    </div>
                    <ul>{items.map((item, itemIndex) => <li key={`${key}-${itemIndex}`}>{item}</li>)}</ul>
                  </article>;
                })}
              </section>
            </div>}
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
