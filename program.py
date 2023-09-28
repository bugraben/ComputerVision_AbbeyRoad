import cv2 as cv
from time import sleep
import numpy as np
from multiprocessing import Process, Manager
from datetime import datetime

BGR_CHANNELS = (0, 1, 2)
channel_names = ["BLUE", "GREEN", "RED"]

date = datetime.now()
fileName = "speed_up.mp4"
ratio = [16,9]
in_width = 1920   
in_height = int(in_width / ratio[0] * ratio[1])
samplingRate = 1
out_width = 1920
out_height= int(out_width / ratio[0] * ratio[1])
sample_count = 160
preview_resolution = (int(in_width*0.35) , int(in_height*0.35))


outputImage = np.zeros((in_height, in_width, 3), dtype = "uint8")

def processChannel(channel = None, frames = None, result = None):
    oneChannelFrame = np.zeros((in_height, in_width), dtype="uint8")
    oneChannelFrame_visualize = np.zeros((in_height, in_width, 3), dtype="uint8")
    if channel in BGR_CHANNELS:
        for row in range(0,in_height):
            print(f"ROW: {row} | CHA: {channel_names[channel]}")
            for column in range(0,in_width):
                pixel = np.array([],dtype="uint8")
                for fr in range(0,sample_count,samplingRate):
                    pixel = np.append(pixel,frames[fr,row,column,channel])
                    
                oneChannelFrame[row, column] = pixel.mean()
            oneChannelFrame_visualize[:, :, channel] = cv.putText(cv.resize(oneChannelFrame, preview_resolution), f"{row}/{in_height}", (50, 50), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (50), 4)
            cv.imshow(channel_names[channel], oneChannelFrame_visualize)
            cv.waitKey(1)
        cv.destroyAllWindows()
        result.append(oneChannelFrame)
                
    else:
        raise ValueError("results: Argument 'channel' must be one of %r." % BGR_CHANNELS)

def captureSequence():
    frames_0 = np.array([], dtype="uint8")
    frames_1 = np.array([], dtype="uint8")
    frames_2 = np.array([], dtype="uint8")
    frames_3 = np.array([], dtype="uint8")
    cap = cv.VideoCapture(fileName)
    for a in range(sample_count):
        ret, frame = cap.read()
        if ret == True:
            # sleep(0.001)
            frame = cv.resize(frame,(in_width,in_height))
            # frames = np.append(frames, frame)

            if a < 400:
                frames_0 = np.append(frames_0, frame)
            elif a < 800:
                frames_2 = np.append(frames_1, frame)
            elif a < 1200:
                frames_2 = np.append(frames_2, frame)
            else:
                frames_3 = np.append(frames_3, frame)

            # print(a)
            cv.imshow("Video", cv.resize(cv.putText(frame, str(a), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255 , 255, 255), 3) , preview_resolution))

        else: 
            print("Not enough samples!")
            break

        if cv.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv.destroyAllWindows()
    print("Closed")

    frames = np.array([frames_0, frames_1, frames_2, frames_3], dtype="uint8")
    frames = frames.reshape(sample_count, in_height, in_width, 3)
    return frames

if __name__ == '__main__':

    captureSequence()


    manager = Manager()

    channel_blue = manager.list()
    channel_green = manager.list()
    channel_red = manager.list()


    p_blue = Process(target=processChannel, args=(0, frames, channel_blue))
    p_green = Process(target=processChannel, args=(1, frames, channel_green))
    p_red = Process(target=processChannel, args=(2, frames, channel_red))


    p_blue.start()
    p_green.start()
    p_red.start()
    print("Processes have been started.")


    p_blue.join()
    p_green.join()
    p_red.join()
    print("Processes OK.")


    outputImage[:, :, 0] = channel_blue[0]
    outputImage[:, :, 1] = channel_green[0]
    outputImage[:, :, 2] = channel_red[0]

    outputRsz = cv.resize(outputImage, (out_width,out_height))
    cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples_{date.day}{date.month}{date.year}.png",outputRsz)

