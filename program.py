import cv2 as cv
from time import sleep
import numpy as np
from multiprocessing import Process, Manager

BGR_CHANNELS = (0, 1, 2)
channel_names = ["BLUE", "GREEN", "RED"]


fileName = "Abbey_Road.mov"
ratio = [16,9]
in_width = 640   
in_height = int(in_width / ratio[0] * ratio[1])
samplingRate = 2
out_width = 960
out_height= int(out_width / ratio[0] * ratio[1])
sample_count = 12

frames = np.array([], dtype="uint8")

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
                oneChannelFrame_visualize[:, :, channel] = oneChannelFrame
            cv.imshow(channel_names[channel], oneChannelFrame_visualize)
            cv.waitKey(1)
        cv.destroyAllWindows()
        result.append(oneChannelFrame)
                
    else:
        raise ValueError("results: Argument 'channel' must be one of %r." % BGR_CHANNELS)





if __name__ == '__main__':
    cap = cv.VideoCapture(fileName)

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
    print("Closed")

    frames = frames.reshape(sample_count, in_height, in_width, 3)
    
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

    p_blue.join()
    p_green.join()
    p_red.join()
    print("Processes OK.")

    finalFrame = np.zeros((in_height, in_width, 3), dtype = "uint8")

    finalFrame[:, :, 0] = channel_blue[0]
    finalFrame[:, :, 1] = channel_green[0]
    finalFrame[:, :, 2] = channel_red[0]

    outputRsz = cv.resize(finalFrame, (out_width,out_height))
    cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples.png",outputRsz)

