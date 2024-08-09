import cv2
import numpy as np
import pickle
import os
import face_recognition


def video_to_data(video_path, name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}.")
        return None

    face_encodings = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 20 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings_in_frame:
                    face_encodings.extend(face_encodings_in_frame)
    cap.release()

    if face_encodings:
        face_encoding = np.mean(face_encodings, axis=0)
        save_user_data(name, video_path, face_encoding)
        print(f"Video and face data saved as {video_path}")
        return True
    else:
        print("No faces detected.")
        return False


def record_video(name, duration=10, video_path='user_video.mp4'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening the camera.")
        return False

    # Set frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

    start_time = cv2.getTickCount()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(duration * fps)

    face_encodings = []

    for _ in range(frame_count):
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings_in_frame:
                    face_encodings.extend(face_encodings_in_frame)

            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if face_encodings:
        face_encoding = np.mean(face_encodings, axis=0)
        save_user_data(name, video_path, face_encoding)
        print(f"Video and face data saved as {video_path}")
        return True
    else:
        print("No faces detected.")
        return False


def save_user_data(name, video_path, face_encoding):
    file_path = 'recognitionvideos/user_data.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_user_data = pickle.load(f)
    else:
        all_user_data = {}

    all_user_data[name] = {'video_path': video_path, 'face_encoding': face_encoding}

    with open(file_path, 'wb') as f:
        pickle.dump(all_user_data, f)
    print("User data saved.")


def load_user_data():
    file_path = 'recognitionvideos/user_data.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_user_data = pickle.load(f)
        return all_user_data
    else:
        print("No user data found.")
        return None


def detect_face_live():
    print("Starting live face recognition.")
    all_user_data = load_user_data()
    if all_user_data is None:
        print("No user data found. Exiting.")
        return

    known_face_encodings = {name: data['face_encoding'] for name, data in all_user_data.items()}
    known_face_names = {name: name for name in known_face_encodings.keys()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening the camera.")
        return

    def process_frame(frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            matches = face_recognition.compare_faces(list(known_face_encodings.values()), encoding)
            name = "Unknown"
            if True in matches:
                matched_idx = matches.index(True)
                name = list(known_face_names.values())[matched_idx]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error reading frame.")
            break

        # Resize the frame to speed up face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        processed_frame = process_frame(small_frame)

        # Check for valid frame before resizing
        if processed_frame is None or processed_frame.size == 0:
            print("Error: Empty or invalid frame.")
            continue

        # Resize back to original size
        processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))

        cv2.imshow('Live Face Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    new_user_data = input("Do you want to record new user data? (y/n): ")
    if new_user_data == 'y':
        user_data = None
    else:
        user_data = load_user_data()

    if user_data is None:
        live_or_video = input(
            "No user data found. Start video processing or live face recognition? (v/l): ")
        if live_or_video == 'v':
            video_path = input("Enter the path to the video: ")
            if video_path:
                name = input("Enter your name: ")
                if name:
                    if video_to_data(video_path, name):
                        print(f"User data for {name} saved and video processed.")
                    else:
                        print("Error processing the video.")
        else:
            name = input("Enter your name: ")
            if name:
                if record_video(name):
                    print(f"User data for {name} saved and video recorded.")
                else:
                    print("Error recording the video.")

    else:
        print(f"User(s) {', '.join(user_data.keys())} found. Starting live face recognition.")
        detect_face_live()


if __name__ == "__main__":
    main()
