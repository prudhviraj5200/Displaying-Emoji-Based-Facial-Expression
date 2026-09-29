from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import os

# Create main window
main = tkinter.Tk()
main.title("Displaying Emoji Based Facial Expressions")
main.geometry("1200x1200")

# Global variables
filename = ""
faces = []
frame = None

# Model paths
detection_model_path = "models/haarcascade_frontalface_default.xml"
emotion_model_path = "models/_mini_XCEPTION.106-0.65.hdf5"

# Load models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Emotion labels
EMOTIONS = [
    "angry",
    "disgust",
    "scared",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# Upload image
def upload():
    global filename

    filename = askopenfilename(initialdir="images")
    pathlabel.config(text=filename)


# Preprocess uploaded image and detect face
def preprocess():
    global filename
    global frame
    global faces

    text.delete("1.0", END)

    # Read image
    orig_frame = cv2.imread(filename)

    if orig_frame is None:
        messagebox.showinfo(
            "Error",
            "Unable to load the selected image."
        )
        return

    # Resize image
    orig_frame = cv2.resize(orig_frame, (48, 48))

    # Convert image to grayscale
    frame = cv2.imread(filename, 0)

    # Detect faces
    faces = face_detection.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    text.insert(
        END,
        "Total number of faces detected : " + str(len(faces))
    )


# Detect facial expression from uploaded image
def detectExpression():
    global faces

    if len(faces) > 0:

        # Select the largest detected face
        faces = sorted(
            faces,
            reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1])
        )[0]

        (fX, fY, fW, fH) = faces

        # Extract face region
        roi = frame[fY:fY + fH, fX:fX + fW]

        # Resize to model input size
        roi = cv2.resize(roi, (48, 48))

        # Normalize pixel values
        roi = roi.astype("float") / 255.0

        # Convert image to array
        roi = img_to_array(roi)

        # Add batch dimension
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = emotion_classifier.predict(roi)[0]

        # Get probability and label
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # Load corresponding emoji
        img = cv2.imread("Emoji/" + label + ".png")

        # Resize emoji
        img = cv2.resize(img, (600, 400))

        # Display detected emotion
        cv2.putText(
            img,
            "Facial Expression Detected As : " + label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Facial Expression Detected As : " + label,
            img
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        messagebox.showinfo(
            "Facial Expression Prediction Screen",
            "No face detected in uploaded image"
        )


# Detect emotion from webcam frame
def detectfromvideo(image):

    result = "none"

    temp = image

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    output = "none"

    if len(faces) > 0:

        # Select largest face
        faces = sorted(
            faces,
            reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1])
        )[0]

        (fX, fY, fW, fH) = faces

        # Extract face
        roi = temp[fY:fY + fH, fX:fX + fW]

        # Convert to grayscale
        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # Resize
        roi = cv2.resize(
            roi,
            (48, 48)
        )

        # Normalize
        roi = roi.astype("float") / 255.0

        # Convert to array
        roi = img_to_array(roi)

        # Add batch dimension
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = emotion_classifier.predict(roi)[0]

        emotion_probability = np.max(preds)

        label = EMOTIONS[preds.argmax()]

        output = label

    return output


# Detect expression from webcam
def detectWebcamExpression():

    cap = cv2.VideoCapture(0)

    while True:

        _, img = cap.read()

        height, width, channels = img.shape

        # Detect emotion
        result = detectfromvideo(img)

        if result != "none":

            print(result)

            # Load emoji
            img1 = cv2.imread(
                "Emoji/" + result + ".png"
            )

            # Resize emoji
            img1 = cv2.resize(
                img1,
                (width, height)
            )

            # Display emotion on emoji
            cv2.putText(
                img1,
                "Facial Expression Detected As : " + result,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Emoji Output",
                img1
            )

            # Display emotion on original webcam frame
            cv2.putText(
                img,
                "Facial Expression Detected As : " + result,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Facial Expression Output",
            img
        )

        # Press Q to exit
        if cv2.waitKey(650) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- GUI ----------------

font = ("times", 20, "bold")

title = Label(
    main,
    text="Displaying Emoji Based Facial Expressions"
)

title.config(
    bg="brown",
    fg="white"
)

title.config(
    font=font,
    height=3,
    width=80
)

title.place(
    x=5,
    y=5
)


font1 = ("times", 14, "bold")


# Upload button
upload_button = Button(
    main,
    text="Upload Image With Face",
    command=upload
)

upload_button.place(
    x=50,
    y=100
)

upload_button.config(
    font=font1
)


# File path label
pathlabel = Label(main)

pathlabel.config(
    bg="brown",
    fg="white",
    font=font1
)

pathlabel.place(
    x=300,
    y=100
)


# Preprocess button
preprocessbutton = Button(
    main,
    text="Preprocess & Detect Face in Image",
    command=preprocess
)

preprocessbutton.place(
    x=50,
    y=150
)

preprocessbutton.config(
    font=font1
)


# Detect expression button
emotion_button = Button(
    main,
    text="Detect Facial Expression",
    command=detectExpression
)

emotion_button.place(
    x=50,
    y=200
)

emotion_button.config(
    font=font1
)


# Webcam button
webcam_button = Button(
    main,
    text="Detect Facial Expression from WebCam",
    command=detectWebcamExpression
)

webcam_button.place(
    x=50,
    y=250
)

webcam_button.config(
    font=font1
)


# Text output area
font1 = ("times", 12, "bold")

text = Text(
    main,
    height=10,
    width=150
)

scroll = Scrollbar(text)

text.configure(
    yscrollcommand=scroll.set
)

text.place(
    x=10,
    y=300
)

text.config(
    font=font1
)


# Background
main.config(
    bg="brown"
)


# Start application
main.mainloop()
