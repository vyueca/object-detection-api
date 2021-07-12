import cv2


def detectObjects(video):
    thres = 0.45  # Threshold to detect object

    classNames = []
    classFile = "object-models/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    configPath = "object-models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "object-models/frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    frameCounter = 0

    while frameCounter <= 100:
        success, img = video.read()

        frameCounter += 1

        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        print(classIds, bbox)

        dictionaryOfObjects = {}

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                confidenceImproved = round(confidence * 100, 2)
                if confidenceImproved > 50:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    if classNames[classId - 1].upper() in dictionaryOfObjects:
                        dictionaryOfObjects[classNames[classId - 1].upper()] += 1
                    else:
                        dictionaryOfObjects[classNames[classId - 1].upper()] = 1

                    print('OBJECT: ')
                    print(classNames[classId - 1].upper())
                    cv2.putText(img, str(confidenceImproved), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)
                    print('VALUE: ')
                    print(str(confidenceImproved))

        cv2.imshow("Output", img)
        cv2.waitKey(1)

        return dictionaryOfObjects
