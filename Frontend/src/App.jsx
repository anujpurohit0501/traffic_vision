import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

// Change this if your Flask backend runs somewhere else
const BACKEND_URL = "http://localhost:5000";

function App() {
  const [mode, setMode] = useState(null); // "webcam" | "upload" | null
  const [streamUrl, setStreamUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const startWebcam = () => {
    setError(null);
    setMode("webcam");
    // cache-bust so the browser always opens a fresh MJPEG connection
    setStreamUrl(`${BACKEND_URL}/video_feed?t=${Date.now()}`);
  };

  const stopWebcam = async () => {
    try {
      await axios.post(`${BACKEND_URL}/stop_feed`);
    } catch (err) {
      console.error(err);
    }
    setMode(null);
    setStreamUrl(null);
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setError(null);
    setUploading(true);
    setMode("upload");

    const formData = new FormData();
    formData.append("video", file);

    try {
      const res = await axios.post(`${BACKEND_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setStreamUrl(`${BACKEND_URL}${res.data.stream_url}`);
    } catch (err) {
      console.error(err);
      setError("Upload failed. Is the Flask backend running on :5000?");
      setMode(null);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Moving Object Detection</h1>
        <p className="subtitle">
          YOLOv7-powered person / vehicle detection &mdash; live webcam or
          uploaded video
        </p>
      </header>

      <div className="controls">
        <button
          className={`btn ${mode === "webcam" ? "btn-active" : ""}`}
          onClick={mode === "webcam" ? stopWebcam : startWebcam}
        >
          {mode === "webcam" ? "Stop Webcam" : "Start Webcam Detection"}
        </button>

        <button
          className="btn"
          onClick={() => fileInputRef.current.click()}
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload Video"}
        </button>
        <input
          type="file"
          accept="video/*"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
      </div>

      {error && <p className="error">{error}</p>}

      <div className="video-panel">
        {streamUrl ? (
          <img src={streamUrl} alt="Detection stream" className="video-feed" />
        ) : (
          <div className="placeholder">
            Choose a source above to start detection
          </div>
        )}
      </div>

      <div className="legend">
        <span className="dot moving" /> moving object
        <span className="dot static" /> static / parked object
      </div>
    </div>
  );
}

export default App;
