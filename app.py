import os
import tempfile
import imghdr
import base64
import random
import matplotlib.pyplot as plt  # Add matplotlib import
from flask import Flask, render_template, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from collections import deque


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
PROCESSED_FOLDER = os.path.join(os.path.dirname(__file__), 'processed_videos')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
my_model = load_model(model_path)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CLASSES_LIST = ["Non-Violence", "Violence"]


def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255.0
    return normalized_frame


def generate_alarm_tone(duration_seconds=1):
    sample_rate = 44100
    frequency = 440
    t = np.linspace(0, duration_seconds, int(
        sample_rate * duration_seconds), endpoint=False)
    alarm_tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    return (alarm_tone * 32767).astype(np.int16)


def get_predicted_frames(video_path):
    predicted_frames = []
    video_reader = cv2.VideoCapture(video_path)

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break
        # Convert frame to base64 for display in HTML
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        predicted_frames.append(frame_base64)

    video_reader.release()
    return predicted_frames


def predict_frames(video_file_path, output_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        ok, frame = video_reader.read()

        if not ok:
            break

        normalized_frame = preprocess_frame(frame)
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = my_model.predict(
                np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
            print("Predicted class name:", predicted_class_name)
            print("Predicted probabilities:", predicted_labels_probabilities)

        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()


def show_pred_frames(pred_video_path, output_image_path):
    plt.figure(figsize=(20, 15))

    video_reader = cv2.VideoCapture(pred_video_path)
    frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    random_range = sorted(random.sample(range(16, frames_count), 12))

    for counter, random_index in enumerate(random_range, 1):
        plt.subplot(5, 4, counter)
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, random_index)
        ok, frame = video_reader.read()

        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.tight_layout()

    video_reader.release()
    plt.savefig(output_image_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        uploaded_file = request.files['video']
        if uploaded_file.filename != '':
            try:
                filename = secure_filename(uploaded_file.filename)
                video_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(video_path)
                output_video_path = os.path.join(
                    PROCESSED_FOLDER, f'processed_{filename}')
                predict_frames(video_path, output_video_path, 16)
                return redirect(f'/result?video={filename}')
            except Exception as e:
                flash(f'Error occurred during prediction: {str(e)}')
                return redirect('/result')
        else:
            flash('Please upload a video file.')
            return redirect('/result')


@app.route('/result')
def result():
    video_name = request.args.get('video', '')
    video_path = os.path.join(PROCESSED_FOLDER, f'processed_{video_name}')
    predicted_frames = get_predicted_frames(video_path)
    output_image_path = os.path.join(PROCESSED_FOLDER, 'prediction_plot.png')
    show_pred_frames(video_path, output_image_path)
    return render_template('result.html', prediction="Violence", video_name=f'processed_{video_name}', predicted_frames=predicted_frames, output_image_path=output_image_path)


@app.route('/processed_videos/<path:filename>')
def processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
