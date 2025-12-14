import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#Load audio
y, sr = librosa.load("data/hum.wav", sr=None)

print(f"Sample rate: {sr}")
print(f"Duration: {len(y)/sr:.2f}s")

#Detect onsets (note/syllable starts)
onsets = librosa.onset.onset_detect(
    y=y,
    sr=sr,
    units="time",
    backtrack=True
)

#Compute durations between onsets
durations = np.diff(onsets)

print("Onsets (s):", np.round(onsets, 3))
print("Durations (s):", np.round(durations, 3))

#Visualization
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(onsets, -1, 1, color="r", linestyle="--")
plt.title("Hummed audio with detected onsets")
plt.show()
