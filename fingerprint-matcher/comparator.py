import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path

def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp1 = temp0 / 255
    filtersize = 6
    W, H = temp0.shape[:2]

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]

            flag = 0
            if np.sum(filter0[:,0]) == 0:
                flag += 1
            if np.sum(filter0[:,filtersize - 1]) == 0:
                flag += 1
            if np.sum(filter0[0,:]) == 0:
                flag += 1
            if np.sum(filter0[filtersize - 1,:]) == 0:
                flag += 1
            if flag > 3:
                temp1[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp1

def get_descriptors(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img[img == 255] = 1

    skeleton = skeletonize(img)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)

    harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125

    keypoints = [cv2.KeyPoint(y, x, 1) for x in range(harris_normalized.shape[0]) for y in range(harris_normalized.shape[1]) if harris_normalized[x][y] > threshold_harris]
    orb = cv2.ORB_create()
    _, des = orb.compute(img, keypoints)
    return keypoints, des

def process_fingerprint(filepath):
    filename = os.path.basename(filepath)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    kp, des = get_descriptors(img)
    print(f"Image: {filename} - Keypoints: {len(kp)} - Descriptors: {des.shape[0]}")
    return filename, des

def save_descriptors(descriptors, db_file):
    with open(db_file, 'wb') as f:
        pickle.dump(descriptors, f)

def load_descriptors(db_file):
    with open(db_file, 'rb') as f:
        return pickle.load(f)

def main():
    fingerprintDatabase = sys.argv[1]
    searchesFingerprint = sys.argv[2]
    db_file = 'fingerprint_database.pkl'

    # Get all files from the database directory
    all_files = [filename for filename in os.listdir(fingerprintDatabase)
                 if os.path.isfile(os.path.join(fingerprintDatabase, filename))]

    # Limit to the first 1000 files
    files_to_process = all_files[:1000]

    for filename in files_to_process:
        print(filename)

    checkFiles = input("Do you want to continue? (y/n): ")
    if checkFiles.lower() != 'y':
        sys.exit()

    # Check if database file exists and load if available
    if Path(db_file).exists():
        input("Database file found, press y to load or any other key to regenerate: ")
        if checkFiles.lower() == 'y':
            descriptors = load_descriptors(db_file)
        else:
            descriptors = []
    else:
        descriptors = []

        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            future_to_filepath = {
                executor.submit(process_fingerprint, os.path.join(fingerprintDatabase, filename)): filename
                for filename in files_to_process
            }

            for future in as_completed(future_to_filepath):
                filename = future_to_filepath[future]
                try:
                    result = future.result()
                    descriptors.append(result)
                except Exception as exc:
                    print(f'{filename} generated an exception: {exc}')

        save_descriptors(descriptors, db_file)

    img = cv2.imread(searchesFingerprint, cv2.IMREAD_GRAYSCALE)
    kp, des = get_descriptors(img)
    print(f"Search Image: {searchesFingerprint} - Keypoints: {len(kp)} - Descriptors: {des.shape[0]}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_descriptor(descriptor):
        filename, des2 = descriptor
        matches = sorted(bf.match(des, des2), key=lambda match: match.distance)
        score = sum(match.distance for match in matches) / len(matches) if matches else float('inf')
        return filename, score

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        scores = list(executor.map(match_descriptor, descriptors))

    scores = sorted(scores, key=lambda score: score[1])

    for score in scores:
        print(f"Image: {score[0]} - Score: {score[1]}")

    best_match = scores[0]
    filepath = os.path.join(fingerprintDatabase, best_match[0])
    if os.path.isfile(filepath):
        img2 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        kp2, des2 = get_descriptors(img2)
        matches = sorted(bf.match(des, des2), key=lambda match: match.distance)
        img3 = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=2)
        plt.imshow(img3)
        plt.show()
    else:
        print("Best match not found (Error in file path)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
