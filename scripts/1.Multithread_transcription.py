'''
This code is to:
 -convert local mp3 and flac files to wav format
 -convert channels from stereo to mono
 -trim the audio length to max 3 mins
 -upload files to google cloud storage bucket
 -do a batch transcription of the audio files
 - write out the transcription to output csv ( provided 'transcription_master.csv')

'''

from google.cloud import storage
from pydub import AudioSegment
import wave
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
import os
import scipy.io.wavfile as wav
import soundfile
import subprocess
import pandas as pd
from multiprocessing import Pool
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool as Pool

'''Access keys to google cloud platform is needed , for which one needs a subscription. If already available, please 
put the key paths in below to positions'''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\MS CS\Spring 2020\ML\SpamCallRecognition\wip\Speech-Recog-speech.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\MS CS\Spring 2020\ML\SpamCallRecognition\wip\speech-recog-storage.json"


''' Below function converts non wav files to wav using ffmpeg 
 If you don't have ffmpeg installed , follow this link :
 https://www.ffmpeg.org/download.html'''

def convertToWav(inpath,outpath):
    print("Checking if file conversion to .wav format is needed")
    for audiofile in os.listdir(inpath):
        if(audiofile.split('.')[1] != 'wav'):
            print("Converting to .wav format...")
            outfile=audiofile.split('.')[0]+'.wav'
            subprocess.call(['ffmpeg', '-i', inpath+"\\"+audiofile,outpath+"\\"+outfile])
        else:
            command='copy "'+inpath+'\\'+audiofile+'" "'+outpath+'\\"'
            print(command)
            os.system(command)
        print("File conversion (if required) completed!")

''' Below function checks for length of audio files, if they are more than 3 minutes, they get trimmed to 3 minutes
    This is done to reduce the computational expense and improve the accuracy of transcription . Since, the longer 
    the audio file , the lesser accurate is the transcription'''

def batch_trim(directory):
    print("If audio duration is more than 3 minutes, it shall be trimmed...")
    for filename in os.listdir(directory):
        input_file_path = os.path.join(directory, filename)
        try:
            freq, data = wav.read(input_file_path)
            duration = len(data) / float(freq)
            if(duration>180.00):
                start = 0
                end = 180
                # output_file_path = input_file_path[:-4]+"-scipy.wav"
                output_file_path = input_file_path
                wav.write(filename=output_file_path, rate=freq, data=data[start*freq : end*freq])
        except:
            print("{} couldn't be trimmed".format(input_file_path))
    print("Trimming operation completed!")

''' Below function checks for the number of channels of audio file. Since google speech recognition tool works on mono
channel only, this function checks and converts channel of a wav file if needed, to mono'''
def stereo_to_mono(directory):
    print("If audio is not mono, converting to mono channel...")
    for audiofile in os.listdir(directory):
        try:
            audio_file_name= os.path.join(directory, audiofile)
            data, samplerate = soundfile.read(audio_file_name)
            soundfile.write(audio_file_name, data, samplerate, subtype='PCM_16')
            with wave.open(audio_file_name, "rb") as wave_file:
                channels = wave_file.getnchannels()
                if channels > 1:
                    sound = AudioSegment.from_wav(audio_file_name)
                    sound = sound.set_channels(1)
                    sound.export(audio_file_name, format="wav")
        except:
            print("{} coudn't be converted to mono".format(audiofile))
    print("Channel conversion completed!")


''' Below function is used to upload objects to google storage . We have used the following code template provided by 
google to create the below code https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python'''
def upload_blob(bucket_name, localpath, file_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    print("Uploading File...")
    source_file_name=os.path.join(localpath, file_name)
    blob.upload_from_filename(source_file_name)
    print("100% completed")
    print("File {} uploaded".format(source_file_name))

''' Below function accepts an audio file uploaded in google storage and transcribes it to english and then returns 
a dictionary with the transcriptuion for each file'''
def speech_recog(gcs_file,dict):
    """
    Transcribe a short audio file using synchronous speech recognition

    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
    """
    client = speech_v1.SpeechClient()

    # The language of the supplied audio
    language_code = "en-US"
    # language_code = "hi-IN"

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    # audio_channel_count = 2
    # enable_separate_recognition_per_channel=2
    # model="phone_call"
    model="default"
    config = {
        "language_code": language_code,
        # "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
        "model": model
        # "audio_channel_count":audio_channel_count,
        # "enable_separate_recognition_per_channel":enable_separate_recognition_per_channel
    }

    audio = {"uri": gcs_file}
    operation = client.long_running_recognize(config, audio)
    print(u"Waiting for transcription operation to complete...")
    response = operation.result()
    output = ""
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        output += alternative.transcript
    dict[gcs_file] = output
    print(u"Transcription operation completed!")


'''Below function deletes an object in a google storage bucket. We have not deleted any files in bucket. You can find 
all audio provided in the storage paths provided in our report'''
def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))

#-------------------
''' User needs to change the paths as required'''
dict_all = {}
bucket_name = "spam-recog-src" # google storage bucket name
parent_src = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\Audio_Set"
local_source = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\source" # mention the source audio file location
local_dest = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\dest\\" # mention the intermediate audio file location
                                                                       # from where the converted wav files will be uploaded


#Below part is to clean an existing directory in windows . Uncomment if required
# command = 'del /f /Q "' + local_source + '\\' + '*.*"'
# os.system(command)
# command = 'del /f /Q "' + local_dest + '\\' + '*.*"'
# os.system(command)
# file_name = "{}.*".format(i)
# command='copy "'+parent_src+'\\'+file_name+'" "'+local_source+'\\"'
# print(command)
# os.system(command)

''' Run the following three functions to convert and process audio files before transcribing'''
convertToWav(local_source, local_dest)
batch_trim(local_dest)
stereo_to_mono(local_dest)


''' Below is transcribe driver function which is used by the multithreaded utility function to
    asynchronously process audio files'''

def transcribe(file_name):
    print("reached")
    upload_blob(bucket_name, local_dest, file_name)
    gcs_file = "gs://spam-recog-src/{}".format(file_name)
    speech_recog(gcs_file, dict_all)
    df = pd.DataFrame(list(dict_all.items()), columns=['key', 'value'])
    text = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\transcription.csv"
    df.to_csv(text, sep=',', mode='a', header=False) # if rows are duplicated, you can remove entire duplicate rows in csv



'''worker function to run each worker thread'''
def worker(item):
    try:
        transcribe(item)
    except:
        print('error with {}'.format(item))

'''main function to run each worker thread on the each file int he local directory'''
def main():
    pool_size = os.cpu_count()  # your "parallelness"
    pool = Pool(pool_size)
    for item in os.listdir(local_dest):
        pool.apply_async(worker, (item,))
    pool.close()
    pool.join()

if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    main()


# delete_blob(bucket_name,file_name)
# print(dict_all)
# df = pd.DataFrame(list(dict_all.items()), columns=['key', 'value'])
# text = "C:\\MS CS\\Spring 2020\\ML\\SpamCallRecognition\\text_data\\spam.csv"
# df.to_csv(text, sep=',', mode='a', header=False)

