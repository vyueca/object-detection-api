from flask import Flask, jsonify
import cv2
from genderagedetector import detectGenderAge
from objectdetector import detectObjects

app = Flask(__name__)


@app.route('/detector/api/realtime/data', methods=['GET'])
def get_data():

    video = cv2.VideoCapture(0)
    video.set(3, 1280)
    video.set(4, 720)
    video.set(10, 70)

    genderAgeData = detectGenderAge()
    objectsData = detectObjects(video)

    return jsonify({'genderAgeData': genderAgeData}, {'objects': objectsData})


if __name__ == '__main__':
    app.run(debug=True)
