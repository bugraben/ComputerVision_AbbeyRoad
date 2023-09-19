import cv2 as cv
from time import sleep
import numpy as np

fileName = "Abbey_Road.mov"
cap = cv.VideoCapture(fileName)

frames = np.array([], dtype="uint8")

ratio = [16,9]
in_width = 640   
in_height = int(in_width / ratio[0] * ratio[1])
samplingRate = 2
out_width=960
out_height= int(out_width / ratio[0] * ratio[1])
sample_count = 300


for a in range(0,sample_count):
    ret, frame = cap.read()
    if ret == True:
        sleep(0.001)
        frame = cv.resize(frame,(in_width,in_height))
        frames = np.append(frames, frame)
        cv.imshow("Video", frame)

    else: break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()



frames = frames.reshape(sample_count,in_height,in_width,3)
outputFrame = np.zeros((in_height,in_width,3), dtype="uint8")

for bgr in range(0,3):
    for row in range(0,in_height):
        for column in range(0,in_width):
            pixel = np.array([],dtype="uint8")
            for fr in range(0,sample_count,samplingRate):
                pixel = np.append(pixel,frames[fr,row,column,bgr])
                print(f"BGR: {bgr} | ROW: {row} | COL: {column} | FRAME: {fr} ")

            outputFrame[row,column,bgr] = pixel.mean()
        cv.imshow("a", outputFrame)
        cv.waitKey(1)   
    # cv.destroyAllWindows()


outputRsz = cv.resize(outputFrame, (out_width,out_height))
cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples.png",outputRsz)