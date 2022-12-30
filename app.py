import streamlit as st
import os
import tempfile
from PIL import Image
from deepface import DeepFace
from deepface.detectors import OpenCvWrapper
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av

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
backend = st.selectbox('Model', backends)
metric = st.selectbox('Model', metrics)

if test_photo:
    placeholder.empty()
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(test_photo.read())

    recognition = DeepFace.find(img_path=temp_file.name,
                                db_path=PHOTO_DIR,
                                enforce_detection=False,
                                distance_metric=metric,
                                detector_backend=backend,
                                model_name=model
                                )
    record = recognition.head(1)
    record_dict = record.to_dict()
    if 0 in record_dict['identity']:
        placeholder.image(record_dict['identity'][0], width=200)


class EmotionPredictor(VideoProcessorBase):

    def __init__(self) -> None:
        # Sign detector
        self.face_detector = FaceDetector()
        self.model = retrieve_model()
        self.queueprediction = []

    def img_convert(self, image):
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def predict(self, image, shape, reshape):

        img_resized = cv2.resize(image, shape).reshape(reshape)
        pred = self.model.predict(img_resized / 255.)[0]
        return emotion[np.argmax(pred)], np.max(pred)

    def find_faces(self, image):
        image2 = image.copy()
        image_face, faces = self.face_detector.findFaces(image)

        # loop over all faces and print them on the video + apply prediction
        for face in faces:
            if face['score'][0] < 0.9:
                continue

            SHAPE = (48, 48)
            RESHAPE = (1, 48, 48, 1)

            xmin = int(face['bbox'][0])
            ymin = int(face['bbox'][1])
            deltax = int(face['bbox'][2])
            deltay = int(face['bbox'][3])

            start_point = (max(0, int(xmin - 0.3 * deltax)), max(0, int(ymin - 0.3 * deltay)))

            end_point = (
            min(image2.shape[1], int(xmin + 1.3 * deltax)), min(image2.shape[0], int(ymin + 1.3 * deltay)))

            im2crop = image2
            im2crop = im2crop[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            im2crop = self.img_convert(im2crop)
            from PIL import Image
            im = Image.fromarray(im2crop)
            im.save("your_file.jpeg")

            prediction, score = self.predict(im2crop, SHAPE, RESHAPE)
            print(prediction, score)
            self.queueprediction.append((prediction, score))

            if len(self.queueprediction) > 20:
                self.queueprediction = self.queueprediction[-20:]
                print(self.queueprediction)

            emotions_dict = {
                'Angry': 0,
                'Disgust': 0,
                'Fear': 0,
                'Happy': 0,
                'Sad': 0,
                'Surprise': 0,
                'Neutral': 0}
            emotions_responses = {
                'Angry': 'Wow chill out',
                'Disgust': 'Eww',
                'Fear': 'TIME TO PANIC',
                'Happy': 'Keep smiling!!',
                'Sad': 'Aww no please do not be sad',
                'Surprise': 'Ahhhhh',
                'Neutral': 'Show me your mood',
                'happytosad': 'Ohh no what happened '}

            for element in self.queueprediction:
                emotions_dict[element[0]] += 1
            print(emotions_dict)

            # #draw emotion on images
            cv2.putText(image2, f'{prediction}', (start_point[0] + 180, start_point[1] + 300),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.9, (255, 255, 255), 2)

            # {str(score)}
            # (start_point[0]-150, start_point[1]-80)

            maxi = 0
            topemotion = 'Angry'
            for key in emotions_dict:
                if emotions_dict[key] > maxi:
                    maxi = emotions_dict[key]
                    topemotion = key

            top_emotions_list = ['neutral', 'neutral']

            if maxi > 15:
                top_emotions_list.append(topemotion)
                top_emotions_list.pop(0)
                if top_emotions_list[-1] == 'Neutral' and top_emotions_list[-2] == 'Happy':
                    topemotion = 'happytosad'

            test = top_emotions_list[1] == 'Neutral' and top_emotions_list[0] == 'Happy'

            cv2.putText(image2, f'{emotions_responses[topemotion]}', (150, 80), cv2.FONT_HERSHEY_DUPLEX,
                        1, (255, 0, 255), 2)

            # draw rectangle arouond face
            cv2.rectangle(image2, start_point, end_point, (255, 255, 255), 2)

        return faces, image2

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        faces, annotated_image = self.find_faces(image)
        return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")


def video_frame_callback(frame):
    if frame:
        frame = frame.to_ndarray(format="bgr24")
        raw_img = frame.copy()
        resolution = frame.shape
        resolution_x = frame.shape[1]
        resolution_y = frame.shape[0]
        print(resolution_x, resolution_y)
        # recognition = DeepFace.find(img,
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
        return av.VideoFrame.from_ndarray(frame, format="bgr24")
    return frame


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
    # video_processor_factory=EmotionPredictor,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
