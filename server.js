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

app.post(
  "/verify",
  upload.fields([
    { name: "uploadedPhoto", maxCount: 1 },
    { name: "capturedPhoto", maxCount: 1 },
  ]),
  async (req, res) => {
    try {
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
const PORT = process.env.PORT || 3001;
app.listen(PORT, () =>
  console.log(`Face verification service running on port ${PORT}`)
);
