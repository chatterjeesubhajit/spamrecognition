import speech_recognition as sr
r=sr.Recognizer()
with sr.AudioFile("C:\\MS CS\\Spring 2020\\ML\\Final Project\\wav files\\test4.wav") as source:
    r.adjust_for_ambient_noise(source)
    audio1=r.record(source,offset=70,duration=10)
    try:
        s=r.recognize_google(audio1)
        print("Text:"+s)
    except Exception as e:
        print("Exception: "+ str(e))
    audio2= r.record(source, offset=80, duration=10)
    try:
        s = r.recognize_google(audio2)
        print("Text:" + s)
    except Exception as e:
        print("Exception: " + str(e))
    audio3 = r.record(source, offset=90, duration=10)
    try:
        s = r.recognize_google(audio3)
        print("Text:" + s)
    except Exception as e:
        print("Exception: " + str(e))


import matplotlib.pyplot as plt
import librosa
audio_path = "C:\\MS CS\\Spring 2020\\ML\\Final Project\\wav files\\test4.wav"
x , sr = librosa.load(audio_path)
# print(type(x), type(sr))
plt.figure(figsize=(14, 5))
import librosa.display
librosa.display.waveplot(x, sr=sr)

#display Spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#If to pring log of frequencies
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()