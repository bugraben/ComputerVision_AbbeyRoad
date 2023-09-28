import cv2 as cv
from time import sleep
import numpy as np
from datetime import datetime
import multiprocessing

BGR_CHANNELS = (0, 1, 2)
channel_names = ["BLUE", "GREEN", "RED"]

datetime = datetime.now()
fileName = "Abbey_Road_2809.mp4"
ratio = [16,9]
in_width = 1920   
in_height = int(in_width / ratio[0] * ratio[1])
samplingRate = 1
out_width = 1920
out_height= int(out_width / ratio[0] * ratio[1])
sample_count = 1
preview_resolution = (int(in_width*0.20) , int(in_height*0.20))

outputImage = np.zeros((in_height, in_width, 3), dtype = "uint8")

def captureSequence():
    frames = np.array([], dtype="uint8")
    frames_temp = np.array([], dtype="uint8")

    cap = cv.VideoCapture(fileName)

    i = 50

    for a in range(sample_count):
        ret, frame = cap.read()
        if ret == True:
            # sleep(0.001)
            if i > 49:
                print(i)
                i = 0
                frames = np.append(frames, frames_temp)
                frames_temp = np.array([], dtype="uint8")

            i += 1
            frame = cv.resize(frame,(in_width,in_height))
            frames_temp = np.append(frames_temp, frame)


            # print(a)
            cv.imshow("Video", cv.resize(cv.putText(frame, str(a), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255 , 255, 255), 3) , (int(in_width*0.40), int(in_height*0.40))))

        else: 
            print("Not enough samples!")
            break

        if cv.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv.destroyAllWindows()
    print("Closed")
    frames = np.append(frames, frames_temp)
    frames = frames.reshape(sample_count, in_height, in_width, 3)
    return frames

def processChannel(channel = None, frames = None, sequence:int = None):

# preview_resolution, in_height, in_width değişkenleri main scope'ta ama local scopun içinden erişebiliyor

    oneChannelFrame = np.zeros((in_height, in_width), dtype="uint8")
    if channel in BGR_CHANNELS:
        for row in range(sequence-int(in_height/4), sequence):
            print(f"ROW: {row} | CHA: {channel_names[channel]}")
            for column in range(0,in_width):
                pixel = np.array([],dtype="uint8")
                for fr in range(0,sample_count,samplingRate):
                    pixel = np.append(pixel,frames[fr,row,column,channel])
                    
                oneChannelFrame[row, column] = pixel.mean()
            oneChannelFrame_visualize = np.zeros((preview_resolution[1], preview_resolution[0], 3), dtype="uint8")
            oneChannelFrame_visualize[:, :, channel] = cv.resize(oneChannelFrame, preview_resolution)
            cv.imshow(channel_names[channel], cv.putText(oneChannelFrame_visualize, f"{row}/{sequence}", (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2))
            cv.waitKey(1)
        cv.destroyAllWindows()
        return oneChannelFrame  
    else:
        raise ValueError("results: Argument 'channel' must be one of %r." % BGR_CHANNELS)



def main():

    frames = captureSequence()

    seq0 = int(in_height/4)
    seq1 = int(in_height/2)
    seq2 = int(in_height/4*3)

    pool = multiprocessing.Pool()
    channel_blue0 = pool.apply_async(processChannel, (0, frames, seq0))
    channel_blue1 = pool.apply_async(processChannel, (0, frames, seq1))
    channel_blue2 = pool.apply_async(processChannel, (0, frames, seq2))
    channel_blue3 = pool.apply_async(processChannel, (0, frames, in_height))

    channel_blue0.wait()
    channel_blue1.wait()
    channel_blue2.wait()
    channel_blue3.wait()
    print("Processes OK.")


    channel_blue0 = channel_blue0.get()
    channel_blue1 = channel_blue1.get()
    channel_blue2 = channel_blue2.get()
    channel_blue3 = channel_blue3.get()


    channel_blue = np.array(channel_blue0, dtype="uint8")
    channel_blue[seq0:seq1, :] = channel_blue1[seq0:seq1, :]
    channel_blue[seq1:seq2, :] = channel_blue2[seq1:seq2, :]
    channel_blue[seq2:, :] = channel_blue3[seq2:, :]


    outputImage[:, :, 0] = channel_blue
    # outputImage[:, :, 1] = channel_green[0]
    # outputImage[:, :, 2] = channel_red[0]
    outputRsz = cv.resize(outputImage, (out_width,out_height))
    cv.imwrite(f"output_{fileName.split('.')[0]}_{sample_count}_samples_{datetime.day}{datetime.month}{datetime.year}_{datetime.hour}{datetime.minute}.png", channel_blue)


if __name__ == '__main__':

    # if ((in_height/4 == in_height//4) and (in_width/4  == in_width//4)):
    #     raise Exception("Resolution values must be exactly divided by 4")

    main()



