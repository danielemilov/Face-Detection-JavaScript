const express = require("express");
const multer = require("multer");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const cors = require("cors");
const path = require("path");

const app = express();

app.use(cors({
  origin: process.env.FRONTEND_URL || 'https://chat-app-client-five-sand.vercel.app',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

process.on('unhandledRejection', (reason, promise) => {
  console.log('Unhandled Rejection at:', promise, 'reason:', reason);
});

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

let modelsLoaded = false;

app.post(
  "/verify",
  upload.fields([
    { name: "uploadedPhoto", maxCount: 1 },
    { name: "capturedPhoto", maxCount: 1 },
  ]),
  async (req, res) => {
    console.log('Verify endpoint hit');
    console.log('Request headers:', req.headers);
    console.log('Request body:', req.body);
    console.log('Request files:', req.files);
    try {
      if (!modelsLoaded) {
        await loadModels();
        modelsLoaded = true;
        console.log("Face-api models loaded");
      }

      console.log("Received verification request");

      if (!req.files.uploadedPhoto || !req.files.capturedPhoto) {
        console.log("Missing one or both photos");
        return res
          .status(400)
          .json({ message: "Both photos are required for verification." });
      }

      const uploadedImg = await canvas.loadImage(
        req.files.uploadedPhoto[0].buffer
      );
      const capturedImg = await canvas.loadImage(
        req.files.capturedPhoto[0].buffer
      );

      console.log("Processing uploaded image");
      const uploadedDetection = await faceapi
        .detectSingleFace(uploadedImg, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      console.log("Processing captured image");
      const capturedDetection = await faceapi
        .detectSingleFace(capturedImg, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!uploadedDetection || !capturedDetection) {
        console.log("No face detected in one or both images");
        return res
          .status(400)
          .json({
            message:
              "No face detected in one or both images. Please ensure both photos contain clear, visible faces.",
          });
      }

      const distance = faceapi.euclideanDistance(
        uploadedDetection.descriptor,
        capturedDetection.descriptor
      );
      const threshold = 0.6; // Adjust this value based on desired strictness
      const isMatch = distance < threshold;

      console.log(
        `Face verification result: isMatch = ${isMatch}, distance = ${distance}`
      );

      res.json({ isMatch, distance });
    } catch (error) {
      console.error("Face verification error:", error);
      res
        .status(500)
        .json({ message: "Error processing images", error: error.message });
    }
  }
);

// Remove the explicit port listening for Vercel deployment
if (process.env.NODE_ENV !== 'production') {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () =>
    console.log(`Face verification service running on port ${PORT}`)
  );
}

module.exports = app;