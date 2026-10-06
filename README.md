FACE-RECOGNITION

A simple face recognition application built using Python and OpenCV that detects and identifies faces from images and live camera input. It stores face data and can be used to build attendance or access control systems.

📸 Features

Detects and recognizes faces using pre-trained models

Face dataset collection and training module

Real-time recognition using webcam

Attendance log support

Outputs face identity results and records for subsequent use

🚀 Installation
1. Clone the Repository
git clone https://github.com/Rajeevkothapally/FACE-RECOGNITION.git
cd FACE-RECOGNITION

2. Create Python Virtual Environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

3. Install Dependencies

Make sure you have Python 3 installed, then run:

pip install -r requirements.txt


If there’s no requirements.txt, install common packages manually:

pip install opencv-python numpy face-recognition

▶️ Usage
1. Prepare Dataset

Add face images to the faces_dataset/ folder. Each person should have a dedicated sub-folder with their name.

Example structure:

faces_dataset/
├── Alice/
│   ├── img1.jpg
│   └── img2.jpg
└── Bob/
    ├── img1.jpg
    └── img2.jpg

2. Train the Model

Run training script (e.g., train.py, or similar logic in the repo) to encode the face images:

python train.py


This will generate face encodings in face_encodings.pkl.

3. Run the App

Start the face recognition app:

python App.py


This will open a webcam window and perform real-time face recognition. Recognized faces will be labeled, and attendance/records logged.

🧠 API Overview

The repo likely contains modules for:

Module/File	Purpose
App.py	Main application entry point
c.py	Face capture or camera handling
t.py	Training utilities (face encodings)
face_encodings.pkl	Serialized face data used for recognition

⚙️ The application uses OpenCV’s face detection and recognition techniques like Haar Cascades and face embeddings.

Note: Ensure all paths and camera indices are updated properly based on your system.

You can drag and drop screenshots into this README by using GitHub’s web editor.

📜 License

This project is licensed under the GPL-3.0 License — see the LICENSE
 file for details.

📝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to open pull requests.
