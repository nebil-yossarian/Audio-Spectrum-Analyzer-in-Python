import parselmouth 
from parselmouth import praat
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
import pandas as pd


# CHUNK = 1024 * 2             # samples per frame
# FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
# CHANNELS = 1                 # single channel for microphone
# RATE = 44100                 # samples per second
vowel_formants_df = pd.read_excel("vowel_formants.xlsx")
data = None
# https://stackoverflow.com/questions/59056786/python-librosa-with-microphone-input
class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paInt16 
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2 * 2
        self.p = None
        self.stream = None

    
    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.CHUNK,
                                  stream_callback=self.callback
                                  )

    def callback(in_data, frame_count, time_info, flag):
        global data
        # global b,a,fulldata #global variables for filter coefficients and array
        data = np.fromstring(in_data, dtype=np.float32)
        # #do whatever with data, in my case I want to hear my data filtered in realtime
        # audio_data = signal.filtfilt(b,a,audio_data,padlen=200).astype(np.float32).tostring()
        # fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
        return (data, pyaudio.paContinue)

    # def read(self):
    #     if not self.stream:
    #         self.start()
    
    #     data = self.stream.read(self.CHUNK)
    #     # convert data to integers, make np array, then offset it by 127
    #     data_int = struct.unpack(str(2 *self.CHUNK) + 'B', data)
    #     # create np array and offset by 128
    #     data_np = np.array(data_int, dtype='b')[::2] + 128
    #     return data_np

    def stop(self):
        self.stream.close()
        self.p.terminate()


def formants(data, sample_rate, num_formants = 2):
    sound = parselmouth.Sound(data, sample_rate)
    f0min=75
    f0max=300
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    # formants = sound.to_formant_burg()
    # print("Formants", formants)
    formants = praat.call(sound, "To Formant (burg)", 0.1, 5, 5000, 0.1, 50)
    # print("Formants", formants)
    num_points = praat.call(pointProcess, "Get number of points")

    formant_array = np.empty((num_formants, num_points))

    for point in range(0, num_points):
        i = point + 1
        t = praat.call(pointProcess, "Get time from index", i)
        for j in range(num_formants):
            formant_array[j,point] = praat.call(formants, "Get value at time", j+1, t, 'Hertz', 'Linear')

    return formant_array

def get_xy(x,y):
    return (-x + 2700,-y+1000)

def plot_vowels(ax):
    for i , row in vowel_formants_df.iterrows():
        
        x, y = get_xy(row["F2"], row["F1"])
        ax.scatter(x, y)
        ax.annotate(row["Vowel"], (x,y + 25))


def main():
    
    fig, ax = plt.subplots(1, figsize=(15, 7))
    # test_point, = ax.scatter(500,500)


    audio = AudioHandler()
    sample_rate = audio.RATE

    while True:
        data = audio.read()
        print("Read data")
        formant_array = formants(data, sample_rate)
        print("Got formants")
        if min(formant_array.shape) == 0:
            continue
        mean_formant = np.nanmean(formant_array, axis = 1)
        print("Shape formant array", formant_array.shape, len(formant_array))
        print(mean_formant)
        f1, f2 = mean_formant
        x,y = get_xy(int(f2),int(f1))

        plot_vowels(ax)
        ax.scatter(x,y)

        # plt.show()

        # ax.annotate("TEST", (x,y + 25))
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    audio.stop()

if __name__ == '__main__':
    main()



