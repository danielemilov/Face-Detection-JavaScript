const express = require("express");
const multer = require("multer");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const cors = require("cors");
const path = require("path");

const app = express();
app.use(cors());

const upload = multer({ storage: multer.memoryStorage() });

// Load face-api models
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk(
    path.join(__dirname, "models")
  );
  await faceapi.nets.faceLandmark68Net.loadFromDisk(
    path.join(__dirname, "models")
  );
  await faceapi.nets.faceRecognitionNet.loadFromDisk(
    path.join(__dirname, "models")
  );
}

loadModels()
  .then(() => {
    console.log("Face-api models loaded");
  })
  .catch((err) => {
    console.error("Error loading face-api models:", err);
  });

app.post("/verify", upload.single("photo"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    const img = await canvas.loadImage(req.file.buffer);
    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      return res.status(400).json({ message: "No face detected in the image" });
    }

    res.json({ faceDescriptor: Array.from(detection.descriptor) });
  } catch (error) {
    console.error("Face verification error:", error);
    res
      .status(500)
      .json({ message: "Error processing image", error: error.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () =>
  console.log(`Face verification service running on port ${PORT}`)
);
