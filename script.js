const video = document.getElementById('video');
const uploadedImage = document.getElementById('uploadedImage');
const capturedImage = document.getElementById('capturedImage');
const fileUpload = document.getElementById('fileUpload');
const captureButton = document.getElementById('captureButton');
const verifyButton = document.getElementById('verifyButton');
const switchCameraButton = document.createElement('button');
switchCameraButton.textContent = 'Switch Camera';
document.body.appendChild(switchCameraButton);

let uploadedImageDescriptor;
let capturedImageDescriptor;
let currentFacingMode = 'user';

// Load models
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models')
]).then(startVideo).catch(err => console.error('Error loading models:', err));

function startVideo() {
    stopVideoStream();
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: currentFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    })
    .then(stream => {
        video.srcObject = stream;
        console.log('Camera started successfully');
        video.play();
    })
    .catch(err => console.error("Error accessing the camera:", err));
}

function stopVideoStream() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
}

switchCameraButton.addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    startVideo();
});

fileUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    uploadedImage.src = URL.createObjectURL(file);
    uploadedImage.onload = processUploadedImage;
});

async function processUploadedImage() {
    console.log('Processing uploaded image');
    const detection = await detectFace(uploadedImage);
    if (detection) {
        console.log('Face detected in uploaded image');
        uploadedImageDescriptor = detection.descriptor;
        drawDetection(uploadedImage, detection);
    } else {
        console.log('No face detected in the uploaded image');
        alert('No face detected in the uploaded image. Please try again.');
    }
}

captureButton.addEventListener('click', async () => {
    console.log('Capture button clicked');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    capturedImage.src = canvas.toDataURL('image/jpeg');
    console.log('Image captured from video');
    
    capturedImage.onload = () => {
        console.log('Captured image loaded');
        retryFaceDetection(3);
    };
});

async function retryFaceDetection(retries) {
    console.log(`Processing captured image, attempt ${4 - retries}`);
    const detection = await detectFace(capturedImage);
    if (detection) {
        console.log('Face detected in captured image');
        capturedImageDescriptor = detection.descriptor;
        drawDetection(capturedImage, detection);
    } else if (retries > 1) {
        console.log('No face detected, retrying...');
        setTimeout(() => retryFaceDetection(retries - 1), 500);
    } else {
        console.log('No face detected in the captured image after all attempts');
        alert('No face detected in the captured image. Please try again.');
    }
}

verifyButton.addEventListener('click', () => {
    if (uploadedImageDescriptor && capturedImageDescriptor) {
        const distance = faceapi.euclideanDistance(uploadedImageDescriptor, capturedImageDescriptor);
        const threshold = 0.6;
        console.log('Face distance:', distance);
        if (distance < threshold) {
            alert('Face Verified! It\'s the same person.');
        } else {
            alert('Face Not Verified. It\'s a different person.');
        }
    } else {
        alert('Please upload an image and capture a face before verifying.');
    }
});

async function detectFace(image) {
    const detectionOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 });
    const detections = await faceapi.detectAllFaces(image, detectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();
    console.log('Detections:', detections);
    return detections[0];
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