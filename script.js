document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('video');
  const videoCanvas = document.getElementById('videoCanvas');
  const uploadedImage = document.getElementById('uploadedImage');
  const capturedImage = document.getElementById('capturedImage');
  const fileUpload = document.getElementById('fileUpload');
  const takePhotoButton = document.getElementById('takePhotoButton');
  const captureButton = document.getElementById('captureButton');
  const verifyButton = document.getElementById('verifyButton');
  const videoContainer = document.getElementById('videoContainer');
  const capturedImageContainer = document.getElementById('capturedImageContainer');
  const guidanceText = document.getElementById('guidanceText');
  const resultMessage = document.getElementById('resultMessage');
  const restartButton = document.getElementById('restartButton');
  const switchCameraButton = document.getElementById('switchCameraButton');
  const loadingBar = document.querySelector('.loading-bar-progress');
  const loadingPercentage = document.querySelector('.loading-percentage');
  const modelLoadingOverlay = document.getElementById('modelLoadingOverlay');

  let uploadedImageDescriptor;
  let capturedImageDescriptor;
  let currentFacingMode = 'user';
  let faceDetectionInterval;

  // Load models
  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models')
  ]).then(() => {
    console.log('Models loaded');
    modelLoadingOverlay.style.display = 'none';
  }).catch(err => {
    console.error('Error loading models:', err);
    alert('Error loading face detection models. Please refresh the page and try again.');
  });

  function showStep(stepNumber) {
    document.querySelectorAll('.step').forEach(step => step.classList.remove('active'));
    document.getElementById(`step${stepNumber}`).classList.add('active');
  }

  fileUpload.addEventListener('change', handleFileUpload);
  takePhotoButton.addEventListener('click', () => {
    showStep(3);
    startVideo();
  });
  captureButton.addEventListener('click', captureImage);
  switchCameraButton.addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    startVideo();
  });
  verifyButton.addEventListener('click', verifyFaces);
  restartButton.addEventListener('click', restartProcess);

  async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async function(e) {
        uploadedImage.src = e.target.result;
        await new Promise(resolve => uploadedImage.onload = resolve);
        try {
          loadingBar.style.width = '0%';
          loadingPercentage.textContent = '0%';
          const detection = await detectFace(uploadedImage, false);
          if (detection) {
            console.log('Face detected in uploaded image');
            uploadedImageDescriptor = detection.descriptor;
            drawDetection(uploadedImage, detection);
            showStep(2);
          } else {
            console.log('No face detected in the uploaded image');
            alert('No face detected in the uploaded image. Please try again.');
          }
        } catch (error) {
          console.error('Error processing uploaded image:', error);
          alert('Error processing the image. Please try again.');
        } finally {
          loadingBar.style.width = '100%';
          loadingPercentage.textContent = '100%';
        }
      };
      reader.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentLoaded = Math.round((event.loaded / event.total) * 100);
          loadingBar.style.width = percentLoaded + '%';
          loadingPercentage.textContent = percentLoaded + '%';
        }
      };
      reader.readAsDataURL(file);
    }
  }

  async function startVideo() {
    stopVideoStream();
    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      await video.play();
      videoCanvas.width = video.videoWidth;
      videoCanvas.height = video.videoHeight;
      onVideoPlaying();
    } catch (err) {
      console.error("Error accessing the camera:", err);
      if (err.name === 'NotAllowedError') {
        alert('Camera access denied. Please grant permission to use the camera.');
      } else if (err.name === 'NotFoundError') {
        alert('No camera found. Please make sure your device has a camera.');
      } else {
        alert('Error accessing the camera. Please check your camera permissions and ensure you\'re using HTTPS or localhost.');
      }
    }
  }

  function onVideoPlaying() {
    if (faceDetectionInterval) {
      clearInterval(faceDetectionInterval);
    }

    faceDetectionInterval = setInterval(async () => {
      try {
        const detection = await detectFace(video, true);
        if (detection) {
          const ctx = videoCanvas.getContext('2d');
          ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
          const resizedDetection = faceapi.resizeResults(detection, { width: video.videoWidth, height: video.videoHeight });
          faceapi.draw.drawDetections(videoCanvas, [resizedDetection]);
        }
      } catch (error) {
        console.error('Error in face detection:', error);
      }
    }, 100);
  }

  function stopVideoStream() {
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
    }
    if (faceDetectionInterval) {
      clearInterval(faceDetectionInterval);
    }
  }

  async function captureImage() {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      
      capturedImage.src = canvas.toDataURL('image/jpeg');
      await new Promise(resolve => capturedImage.onload = resolve);
      
      const detection = await detectFace(capturedImage, false);
      if (detection) {
        console.log('Face detected in captured image');
        capturedImageDescriptor = detection.descriptor;
        drawDetection(capturedImage, detection);
        showStep(4);
        stopVideoStream();
      } else {
        console.log('No face detected in the captured image');
        alert('No face detected in the captured image. Please try again.');
      }
    } catch (error) {
      console.error('Error capturing image:', error);
      alert('Error capturing image. Please try again.');
    }
  }

  async function detectFace(image, isLiveDetection = true) {
    try {
      const detectionOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 });
      const detections = await faceapi.detectAllFaces(image, detectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();
      console.log('Detections:', detections);
      
      if (detections.length > 0) {
        const detection = detections[0];
        if (isLiveDetection) {
          const isWithinOval = checkIfWithinOval(detection.detection.box, image.videoWidth || image.width, image.videoHeight || image.height);
          guidanceText.textContent = isWithinOval ? 'Perfect! Click capture' : 'Move closer and center your face';
        }
        return detection;
      }
      
      if (isLiveDetection) {
        guidanceText.textContent = 'No face detected. Please try again.';
      }
      return null;
    } catch (error) {
      console.error('Error in face detection:', error);
      throw error;
    }
  }

  function checkIfWithinOval(box, imageWidth, imageHeight) {
    const ovalCenterX = imageWidth / 2;
    const ovalCenterY = imageHeight / 2;
    const ovalWidth = imageWidth * 0.7;
    const ovalHeight = imageHeight * 0.9;
    
    const faceCenterX = box.x + box.width / 2;
    const faceCenterY = box.y + box.height / 2;
    
    const distanceX = Math.abs(faceCenterX - ovalCenterX) / (ovalWidth / 2);
    const distanceY = Math.abs(faceCenterY - ovalCenterY) / (ovalHeight / 2);
    
    return (distanceX * distanceX + distanceY * distanceY) <= 1;
  }

  function drawDetection(image, detection) {
    const displaySize = { width: image.width, height: image.height };
    let canvas = image.nextElementSibling;
    if (!canvas || canvas.tagName !== 'CANVAS') {
      canvas = faceapi.createCanvasFromMedia(image);
      image.parentElement.appendChild(canvas);
    }
    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    
    faceapi.matchDimensions(canvas, displaySize);
    const resizedDetection = faceapi.resizeResults(detection, displaySize);
    faceapi.draw.drawDetections(canvas, [resizedDetection]);
    faceapi.draw.drawFaceLandmarks(canvas, [resizedDetection]);
  }

  function verifyFaces() {
    if (uploadedImageDescriptor && capturedImageDescriptor) {
      const distance = faceapi.euclideanDistance(uploadedImageDescriptor, capturedImageDescriptor);
      const threshold = 0.6;
      console.log('Face distance:', distance);
      const isVerified = distance < threshold;
      
      resultMessage.textContent = isVerified 
        ? 'Face Verified! It\'s the same person.' 
        : 'Face Not Verified. It\'s a different person.';
      resultMessage.style.backgroundColor = isVerified ? '#2ecc71' : '#e74c3c';
      
      showStep(5);
    } else {
      alert('Please upload an image and capture a face before verifying.');
    }
  }

  function restartProcess() {
    uploadedImageDescriptor = null;
    capturedImageDescriptor = null;
    uploadedImage.src = '';
    capturedImage.src = '';
    showStep(1);
    stopVideoStream();
  }

  // Initialize
  showStep(1);

  // Global error handler
  window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    alert('An unexpected error occurred. Please try again.');
  });
});