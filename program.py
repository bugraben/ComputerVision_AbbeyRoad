# Code by bugraben
# This program aims to create clear images of crowded streets using videos.

import cv2 as cv
from time import sleep
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from os import getpid

BGR_CHANNELS = (0, 1, 2)
channel_names = ["BLUE", "GREEN", "RED"]

# Defining input and output parameters.
fileName = "Abbey_Road_2809.mp4"
ratio = [16,9]
in_width = 768   
out_width = 1280

in_height = int(in_width / ratio[0] * ratio[1])
out_height= int(out_width / ratio[0] * ratio[1])
preview_resolution = (480 , 270)

# Abbey_Road_2809.mp4 has 1660 frames of total
sample_count = 1660 
samplingRate = 2

seq_length:int = int(in_height/10)
seq0:int = int(seq_length)
seq1:int = int(seq_length*2)
seq2:int = int(seq_length*3)
seq3:int = int(seq_length*4)
seq4:int = int(seq_length*5)
seq5:int = int(seq_length*6)
seq6:int = int(seq_length*7)
seq7:int = int(seq_length*8)
seq8:int = int(seq_length*9)
seq9:int = int(in_height)

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
            cv.imshow("Video", cv.resize(cv.putText(frame, str(a), (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (255 , 255, 255), 3) , preview_resolution))

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

def processChannel(channel = None, frames = None, sequence:int = 0):

# preview_resolution, in_height, in_width değişkenleri main scope'ta ama local scopun içinden erişebiliyor?

    oneChannelFrame = np.zeros((in_height, in_width), dtype="uint8")
    if channel in BGR_CHANNELS:
        print(getpid())
        for row in range(int(sequence-seq_length), int(sequence)):
            # print(f"ROW: {row} | CHA: {channel_names[channel]}")
            for column in range(0,in_width):
                pixel = np.array([],dtype="uint8")
                for fr in range(0,sample_count,samplingRate):
                    pixel = np.append(pixel,frames[fr,row,column,channel])
                    
                oneChannelFrame[row, column] = pixel.mean()
            if getpid() < 11000:
                oneChannelFrame_visualize = np.zeros((preview_resolution[1], preview_resolution[0], 3), dtype="uint8")
                oneChannelFrame_visualize[:, :, channel] = cv.resize(oneChannelFrame, preview_resolution)
                cv.imshow(channel_names[channel], cv.putText(oneChannelFrame_visualize, f"{row}/{sequence}", (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2))
                cv.waitKey(1)
        cv.destroyAllWindows()
        return oneChannelFrame  
    else:
        raise ValueError("results: Argument 'channel' must be one of %r." % BGR_CHANNELS)

def progressBar():
    global progress
    progress += 1
    print("[", end="", flush=True)
    for i in range(progress):
        print("|", end="", flush=True)
    print("]", end="", flush=True)
    print(f"{progress}/30")


def main():

# Importing the video to an numpy array.
    frames = captureSequence()

    pool = Pool()

# Splitting the frame process as 5 rows and as color channels.
    channel_blue0 = pool.apply_async(processChannel, (0, frames, seq0))
    channel_blue1 = pool.apply_async(processChannel, (0, frames, seq1))
    channel_blue2 = pool.apply_async(processChannel, (0, frames, seq2))
    channel_blue3 = pool.apply_async(processChannel, (0, frames, seq3))
    channel_blue4 = pool.apply_async(processChannel, (0, frames, seq4))
    channel_blue5 = pool.apply_async(processChannel, (0, frames, seq5))
    channel_blue6 = pool.apply_async(processChannel, (0, frames, seq6))
    channel_blue7 = pool.apply_async(processChannel, (0, frames, seq7))
    channel_blue8 = pool.apply_async(processChannel, (0, frames, seq8))
    channel_blue9 = pool.apply_async(processChannel, (0, frames, seq9))
    print("Team blue ready!")

    channel_green0 = pool.apply_async(processChannel, (1, frames, seq0))
    channel_green1 = pool.apply_async(processChannel, (1, frames, seq1))
    channel_green2 = pool.apply_async(processChannel, (1, frames, seq2))
    channel_green3 = pool.apply_async(processChannel, (1, frames, seq3))
    channel_green4 = pool.apply_async(processChannel, (1, frames, seq4))
    channel_green5 = pool.apply_async(processChannel, (1, frames, seq5))
    channel_green6 = pool.apply_async(processChannel, (1, frames, seq6))
    channel_green7 = pool.apply_async(processChannel, (1, frames, seq7))
    channel_green8 = pool.apply_async(processChannel, (1, frames, seq8))
    channel_green9 = pool.apply_async(processChannel, (1, frames, seq9))
    print("Team green ready!")


    channel_red0 = pool.apply_async(processChannel, (2, frames, seq0))
    channel_red1 = pool.apply_async(processChannel, (2, frames, seq1))
    channel_red2 = pool.apply_async(processChannel, (2, frames, seq2))
    channel_red3 = pool.apply_async(processChannel, (2, frames, seq3))
    channel_red4 = pool.apply_async(processChannel, (2, frames, seq4))
    channel_red5 = pool.apply_async(processChannel, (2, frames, seq5))
    channel_red6 = pool.apply_async(processChannel, (2, frames, seq6))
    channel_red7 = pool.apply_async(processChannel, (2, frames, seq7))
    channel_red8 = pool.apply_async(processChannel, (2, frames, seq8))
    channel_red9 = pool.apply_async(processChannel, (2, frames, seq9))
    print("Team red ready!")

# Visualizing the progress.
    global progress
    progress=0
    channel_blue0.wait()
    progressBar() 
    channel_blue1.wait()
    progressBar() 
    channel_blue2.wait()
    progressBar() 
    channel_blue3.wait()
    progressBar() 
    channel_blue4.wait()
    progressBar() 
    channel_blue5.wait()
    progressBar() 
    channel_blue6.wait()
    progressBar() 
    channel_blue7.wait()
    progressBar() 
    channel_blue8.wait()
    progressBar() 
    channel_blue9.wait()
    progressBar()

    channel_green0.wait()
    progressBar() 
    channel_green1.wait()
    progressBar() 
    channel_green2.wait()
    progressBar() 
    channel_green3.wait()
    progressBar() 
    channel_green4.wait()
    progressBar() 
    channel_green5.wait()
    progressBar() 
    channel_green6.wait()
    progressBar() 
    channel_green7.wait()
    progressBar() 
    channel_green8.wait()
    progressBar() 
    channel_green9.wait()
    progressBar()

    channel_red0.wait()
    progressBar() 
    channel_red1.wait()
    progressBar() 
    channel_red2.wait()
    progressBar() 
    channel_red3.wait()
    progressBar() 
    channel_red4.wait()
    progressBar() 
    channel_red5.wait()
    progressBar() 
    channel_red6.wait()
    progressBar() 
    channel_red7.wait()
    progressBar() 
    channel_red8.wait()
    progressBar() 
    channel_red9.wait()
    progressBar()

    print("Progress OK.")

# Importing sequences from sub-processes
    channel_blue0 = channel_blue0.get()
    channel_blue1 = channel_blue1.get()
    channel_blue2 = channel_blue2.get()
    channel_blue3 = channel_blue3.get()
    channel_blue4 = channel_blue4.get()
    channel_blue5 = channel_blue5.get()
    channel_blue6 = channel_blue6.get()
    channel_blue7 = channel_blue7.get()
    channel_blue8 = channel_blue8.get()
    channel_blue9 = channel_blue9.get()

    channel_green0 = channel_green0.get()
    channel_green1 = channel_green1.get()
    channel_green2 = channel_green2.get()
    channel_green3 = channel_green3.get()
    channel_green4 = channel_green4.get()
    channel_green5 = channel_green5.get()
    channel_green6 = channel_green6.get()
    channel_green7 = channel_green7.get()
    channel_green8 = channel_green8.get()
    channel_green9 = channel_green9.get()

    channel_red0 = channel_red0.get()
    channel_red1 = channel_red1.get()
    channel_red2 = channel_red2.get()
    channel_red3 = channel_red3.get()
    channel_red4 = channel_red4.get()
    channel_red5 = channel_red5.get()
    channel_red6 = channel_red6.get()
    channel_red7 = channel_red7.get()
    channel_red8 = channel_red8.get()
    channel_red9 = channel_red9.get()

# Writing sequences into a main frame.
    channel_blue = np.array(channel_blue0, dtype="uint8")
    channel_blue[seq0:seq1, :] = channel_blue1[seq0:seq1, :]
    channel_blue[seq1:seq2, :] = channel_blue2[seq1:seq2, :]
    channel_blue[seq2:seq3, :] = channel_blue3[seq2:seq3, :]
    channel_blue[seq3:seq4, :] = channel_blue4[seq3:seq4, :]
    channel_blue[seq4:seq5, :] = channel_blue5[seq4:seq5, :]
    channel_blue[seq5:seq6, :] = channel_blue6[seq5:seq6, :]
    channel_blue[seq6:seq7, :] = channel_blue7[seq6:seq7, :]
    channel_blue[seq7:seq8, :] = channel_blue8[seq7:seq8, :]
    channel_blue[seq8:seq9, :] = channel_blue9[seq8:seq9, :]

    channel_green = np.array(channel_green0, dtype="uint8")
    channel_green[seq0:seq1, :] = channel_green1[seq0:seq1, :]
    channel_green[seq1:seq2, :] = channel_green2[seq1:seq2, :]
    channel_green[seq2:seq3, :] = channel_green3[seq2:seq3, :]
    channel_green[seq3:seq4, :] = channel_green4[seq3:seq4, :]
    channel_green[seq4:seq5, :] = channel_green5[seq4:seq5, :]
    channel_green[seq5:seq6, :] = channel_green6[seq5:seq6, :]
    channel_green[seq6:seq7, :] = channel_green7[seq6:seq7, :]
    channel_green[seq7:seq8, :] = channel_green8[seq7:seq8, :]
    channel_green[seq8:seq9, :] = channel_green9[seq8:seq9, :]

    channel_red = np.array(channel_red0, dtype="uint8")
    channel_red[seq0:seq1, :] = channel_red1[seq0:seq1, :]
    channel_red[seq1:seq2, :] = channel_red2[seq1:seq2, :]
    channel_red[seq2:seq3, :] = channel_red3[seq2:seq3, :]
    channel_red[seq3:seq4, :] = channel_red4[seq3:seq4, :]
    channel_red[seq4:seq5, :] = channel_red5[seq4:seq5, :]
    channel_red[seq5:seq6, :] = channel_red6[seq5:seq6, :]
    channel_red[seq6:seq7, :] = channel_red7[seq6:seq7, :]
    channel_red[seq7:seq8, :] = channel_red8[seq7:seq8, :]
    channel_red[seq8:seq9, :] = channel_red9[seq8:seq9, :]


    outputImage[:, :, 0] = channel_blue
    outputImage[:, :, 1] = channel_green
    outputImage[:, :, 2] = channel_red

# Resizing output image and exporting.
    outputRsz = cv.resize(outputImage, (out_width,out_height))
    global datetime
    datetime = datetime.now()
    print(f"output_{fileName.split('.')[0]}_{int(sample_count/samplingRate)}_samples_{datetime.day}{datetime.month}{datetime.year}_{datetime.hour}{datetime.minute}.png")
    cv.imwrite(f"output_{fileName.split('.')[0]}_{int(sample_count/samplingRate)}samples_{datetime.day}{datetime.month}{datetime.year}_{datetime.hour}{datetime.minute}.png", outputRsz)


if __name__ == '__main__':

    # if ((in_height/4 == in_height//4) and (in_width/4  == in_width//4)):
    #     raise Exception("Resolution values must be exactly divided by 4")

    main()



