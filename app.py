import streamlit as st
import os
import tempfile
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av
import cv2
from deepface.detectors import FaceDetector

APP_DIR = os.path.abspath(os.curdir)
PHOTO_DIR = os.path.join(APP_DIR, "photos")

models = [
    "Facenet512",
    "Facenet",
    "VGG-Face",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
]

metrics = ["euclidean", "cosine", "euclidean_l2"]

st.set_page_config(
    page_title="Prevantec - Facial Recognition", initial_sidebar_state="expanded"
)

if not os.path.exists(PHOTO_DIR):
    os.mkdir(PHOTO_DIR)


def face_entry(img_path, name_text):
    if img_path == "" or name_text == "" or img_path is None or name_text is None:
        return None
    image = Image.open(img_path)
    rgb_im = image.convert('RGB')
    file_name = os.path.join(PHOTO_DIR, f"{name_text}.jpg")
    rgb_im.save(file_name)
    try:
        os.remove(os.path.join(PHOTO_DIR, "representations_facenet512.pkl"))
    except:
        pass


def show_images(image_data):
    for image_path, name in image_data:
        st.sidebar.image(image_path, caption=name, width=200)


placeholder = st.empty()

person_photo = st.sidebar.file_uploader("Photo", type=[".jpg", ".png", ".jpeg"])
person_name = st.sidebar.text_input("Name")
register_button = st.sidebar.button("Register")

if register_button and person_photo and person_name:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(person_photo.read())
    face_entry(temp_file.name, person_name)

db_photos = os.listdir(PHOTO_DIR)
person_list = [[os.path.join(PHOTO_DIR, i), i.split('.')[0]] for i in db_photos if 'jpg' in i]
show_images(person_list)

placeholder.title("Facial recognition")

test_photo = st.file_uploader("Upload photo", type=[".jpg", ".png", ".jpeg"])

model = st.selectbox('Model', models)
backend = st.selectbox('backend', backends)
metric = st.selectbox('metric', metrics)

if test_photo:
    placeholder.empty()
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(test_photo.read())

    # recognition = DeepFace.find(img_path=temp_file.name,
    #                             db_path=PHOTO_DIR,
    #                             enforce_detection=False,
    #                             distance_metric=metric,
    #                             detector_backend=backend,
    #                             model_name=model
    #                             )
    # record = recognition.head(1)
    # record_dict = record.to_dict()
    # if 0 in record_dict['identity']:
    #     placeholder.image(record_dict['identity'][0], width=200)


class FaceRecognition(VideoProcessorBase):

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        detector = FaceDetector.build_model(backend)
        image = frame.to_ndarray(format="rgb24")
        cv2.putText(image, backend, (int(40), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        faces_detected = FaceDetector.detect_faces(detector, backend, image, align=False)
        for face in faces_detected:
            x, y, w, h = face[1]
            cv2.rectangle(image, (x, y), (int(x + w), int(y + h)), (0, 0, 0), 2)
        return av.VideoFrame.from_ndarray(image, format="rgb24")


# def video_frame_callback(frame):
#     if frame:
#         frame = frame.to_ndarray(format="bgr24")
#         raw_img = frame.copy()
#         resolution = frame.shape
#         resolution_x = frame.shape[1]
#         resolution_y = frame.shape[0]
#         print(resolution_x, resolution_y)
#         # recognition = DeepFace.find(img,
#         #                             db_path=PHOTO_DIR,
#         #                             enforce_detection=False,
#         #                             distance_metric=metric,
#         #                             detector_backend=backend,
#         #                             model_name=model
#         #                             )
#         # record = recognition.head(1)
#         # record_dict = record.to_dict()
#         # if 0 in record_dict['identity']:
#         #     placeholder.image(record_dict['identity'][0], width=200)
#         return av.VideoFrame.from_ndarray(frame, format="bgr24")
#     return frame


# webrtc_streamer(
#     key="camera",
#     video_frame_callback=video_frame_callback,
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={"video": True, "audio": True})
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FaceRecognition,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
