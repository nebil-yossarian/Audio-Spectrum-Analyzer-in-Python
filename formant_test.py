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

y, sample_rate = librosa.load("Close_back_rounded_vowel.ogg")
start_time = time.time()
formants = estimate_formants_lpc(y, sample_rate)
end_time = time.time()
print("Elasped Time:", end_time - start_time)
vowel_formants_df = pd.read_excel("vowel_formants.xlsx")

# plt.scatter(vowel_formants_df["F1"], vowel_formants_df["F2"])
for i , row in vowel_formants_df.iterrows():
    x = -row["F2"] + 2700
    y = -row["F1"] + 1000
    plt.scatter(x, y)
    plt.annotate(row["Vowel"], (x,y + 25))

x = -formants[3] + 2700
y = -formants[2] + 1000
print(formants[2:5])
plt.scatter(x, y)
plt.annotate("TEST", (x,y + 25))
plt.show()
# for i in range(1, 13):
#     print(sample_rate)
#     formants = estimate_formants_lpc(y, sample_rate, num_formants = i)
#     print(i,formants)