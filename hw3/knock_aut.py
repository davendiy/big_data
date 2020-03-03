#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sounddevice as sd
import numpy as np
from scipy.stats import pearsonr
from scipy.io.wavfile import write
import librosa
import os
import matplotlib.pyplot as plt



def get_recording(seconds=5, from_file=None, show=False):
    if from_file:
        y, sr = librosa.load(from_file)
    else:
        fs = 44100  # Sample rate
        print(f"[*] Recording for {seconds}s...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        print('[*] Finished.')
        write('__tmp_file.waw', fs, myrecording)  # Save as WAV file
        y, sr = librosa.load('__tmp_file.waw')
        os.remove('__tmp_file.waw')

    if show:
        plt.plot(y, linewidth=.2)
        plt.show()

    _, tmp_beat = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(tmp_beat, sr=sr)


def check_access(given, pattern, conf_v=0.95):
    n = len(given)
    m = len(pattern)
    if n < m:
        return False, None

    res = False
    max_corr = 0
    for i in range(n-m):
        tmp_corr = pearsonr(given[i:i+m], pattern)[0]
        max_corr = max(max_corr, tmp_corr)
        if tmp_corr > conf_v:
            res = True
            break
    return res, max_corr


def show(signal):
    plt.plot(signal, linewidth=.2)
    plt.show()


if __name__ == '__main__':
    input("Press enter to start the recording of pattern.")
    target_beats = get_recording(show=True)
    print("Finished.", target_beats)
    while True:
        input('Press enter to start the recording and check the access.')
        cur_beats = get_recording(show=True)
        print("Finished", cur_beats)
        print("check access:", check_access(cur_beats, target_beats))
