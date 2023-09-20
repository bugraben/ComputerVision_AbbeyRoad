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
sample_count = 10

frames = np.array([], dtype="uint8")
outputFrame = np.zeros((in_height, in_width), dtype="uint8")

def processChannel(channel = None, frames = None, result = None):
    print(f"Process {channel} has started.")
    if channel in BGR_CHANNELS:
        for row in range(0,in_height):
            print(f"ROW: {row} | CHA: {channel_names[channel]}")
            for column in range(0,in_width):
                pixel = np.array([],dtype="uint8")
                for fr in range(0,sample_count,samplingRate):
                    pixel = np.append(pixel,frames[fr,row,column,channel])
                    
                global outputFrame
                outputFrame[row, column] = pixel.mean()
        result.append(outputFrame)
                
    else:
        raise ValueError("results: Argument 'channel' must be one of %r." % BGR_CHANNELS)

# cv.imshow(fileName, outputFrame)
# cv.waitKey(500)
# def monitorOutput():
#         cv.imshow(fileName, outputFrame)
#         cv.waitKey(1)


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

    frames = frames.reshape(sample_count,in_height,in_width,3)
    
    manager = Manager()

    channel_blue = manager.list()


    p_blue = Process(target=processChannel, args=(0, frames, channel_blue))
    # p_green = Process(target=processChannel, args=(1, frames,))
    # p_red = Process(target=processChannel, args=(2, frames,))
    # p_monitor = Process(target=monitorOutput)

    # p_blue.start()

    p_blue.start()
    # channel_green = p_green.start()
    # channel_red = p_red.start()

    p_blue.join()
    print("Process OK.")
    # p_green.join()
    # p_red.join()


    channel_blue = np.array(channel_blue, dtype="uint8")
    channel_blue = channel_blue.reshape(in_height,in_width)

    print(type(np.array(channel_blue)))

    # cv.imshow(fileName, channel_blue)
    # cv.waitKey(1)
    # cv.imshow(fileName, channel_green)
    # cv.waitKey(1)
    # cv.imshow(fileName, channel_red)
    # cv.waitKey(1)
    # sleep(5000)

    # while p_blue.is_alive() or p_green.is_alive() or p_red.is_alive():
    #     p_monitor.start()
    #     p_monitor.join()
    # else:
    #     cv.destroyAllWindows()

    print(channel_blue)

    cv.imwrite("BLUE.png",channel_blue)
    # cv.imwrite("GREEN.png",channel_green)
    # cv.imwrite("RED.png",channel_red)



    # outputRsz = cv.resize(outputFrame, (out_width,out_height))
    # cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples.png",outputRsz)

