import time
import cv2 as cv
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackCon=0.5):  # default initializations
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.tackCon = trackCon
        # initializations
        self.comlexity = complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.comlexity, self.detectionCon, self.tackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return image

    def findPosition(self,image, handNo=0, draw=True):
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo]
            for idNumber, landmarkInformation in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(landmarkInformation.x * w), int(landmarkInformation.y * h)
                # print(idNumber, cx, cy)
                self.lmList.append([idNumber,cx,cy])

                # if idNumber == 4:
                if draw:
                    if idNumber == 0:
                        cv.circle(image, (cx, cy), 25, (5, 5, 5), cv.FILLED)
                    else:
                        cv.circle(image, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers =[]

        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        return fingers






def main():
    previousTime = 0
    currentTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, image = cap.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image)
        if len(lmList)!=0:
            print(lmList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv.putText(image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv.imshow("Image", image)
        cv.waitKey(1)


if __name__ == '__main__':
    main()