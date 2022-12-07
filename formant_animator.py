import pyaudio
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

from tkinter import TclError
from scipy.signal import lfilter
import librosa

# https://stackoverflow.com/questions/59056786/python-librosa-with-microphone-input
class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2 * 2
        self.p = None
        self.stream = None
        self.data = None

    
    def start(self):
        self.p = pyaudio.PyAudio()
        def callback(in_data, frame_count, time_info, flag):
            # global b,a,fulldata #global variables for filter coefficients and array
            self.data = np.fromstring(in_data, dtype=np.float32)
            # #do whatever with data, in my case I want to hear my data filtered in realtime
            # audio_data = signal.filtfilt(b,a,audio_data,padlen=200).astype(np.float32).tostring()
            # fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
            return (self.data, pyaudio.paContinue)
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  frames_per_buffer=self.CHUNK,
                                  stream_callback=callback)


    def stop(self):
        self.stream.close()
        self.p.terminate()


def estimate_formants_lpc(waveform, sample_rate):
    hamming_win = np.hamming(len(waveform))
    # Apply window and high pass filter.
    x_win = waveform * hamming_win
    x_filt = lfilter([1], [1.0, 0.63], x_win)

    # Get LPC. From mathworks link above, the general rule is that the
    # order is two times the expected number of formants plus 2. We use
    # 5 as a base because we discard the formant f0 and want f1...f4.
    # print(2 + int(sample_rate / 1000))
    lpc_rep = librosa.lpc(x_filt, order = 2 + int(sample_rate / 1000))
    # Calculate the frequencies.
    roots = [r for r in np.roots(lpc_rep) if np.imag(r) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))

    return sorted(angles * (sample_rate / (2 * np.pi)))


def formant_generator(audio):
    audio.start()
    while audio.stream.is_active():
        if audio.data is None:
            continue
        formants =  estimate_formants_lpc(audio.data, audio.RATE)
        yield formants
        # time.sleep(0.2)

audio = AudioHandler()
for formant in formant_generator(audio):
    print(formant[0:5])