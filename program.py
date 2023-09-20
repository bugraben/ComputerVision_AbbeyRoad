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

def processChannel(channel = None, frames = None, result = None):
    oneChannelFrame = np.zeros((in_height, in_width), dtype="uint8")
    if channel in BGR_CHANNELS:
        for row in range(0,in_height):
            print(f"ROW: {row} | CHA: {channel_names[channel]}")
            for column in range(0,in_width):
                pixel = np.array([],dtype="uint8")
                for fr in range(0,sample_count,samplingRate):
                    pixel = np.append(pixel,frames[fr,row,column,channel])
                    
                oneChannelFrame[row, column] = pixel.mean()
        result.append(oneChannelFrame)
                
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
    channel_green = manager.list()
    channel_red = manager.list()


    p_blue = Process(target=processChannel, args=(0, frames, channel_blue))
    p_green = Process(target=processChannel, args=(1, frames, channel_green))
    p_red = Process(target=processChannel, args=(2, frames, channel_red))
    # p_monitor = Process(target=monitorOutput)


    p_blue.start()
    p_green.start()
    p_red.start()

    p_blue.join()
    p_green.join()
    p_red.join()
    print("Processes OK.")


    finalFrame = np.array([channel_blue[0], channel_green[0], channel_red[0]], dtype="uint8")

    finalFrame = finalFrame.reshape(in_height,in_width)

    print(finalFrame)

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


    # cv.imwrite(".png",channel_blue)
    # cv.imwrite("GREEN.png",channel_green)
    # cv.imwrite("RED.png",channel_red)



    outputRsz = cv.resize(finalFrame, (out_width,out_height))
    cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples.png",outputRsz)

