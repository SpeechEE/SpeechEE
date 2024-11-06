from transformers import pipeline
import os
import json
from glob import glob
import torch
import librosa
import jsonlines
modelPath = "xxx/model_cache/whisper-large-v2"


def find_files(directory, pattern='**/*.wav'): 
    return glob(os.path.join(directory, pattern), recursive=True)


datatype = "test"
datasetDir = "PHEE"
sampling_rate =16000
path = os.path.join("xxx","output", datasetDir, datatype)
savePath =os.path.join("xxx","xxx/transcripts", datasetDir)

WavList = find_files(path)
wavPath = [] 
for i in range(len(WavList)):
    wavPath.append(os.path.join(path,datatype+"-"+str(i)+".wav"))



device = "cuda:0" if torch.cuda.is_available() else "cpu"
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# load model and processor


model = WhisperForConditionalGeneration.from_pretrained(modelPath).to(device)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(modelPath)
inputAudios = [] 
for i in range(len(wavPath)):

        audio, _ = librosa.load(wavPath[i], sr=sampling_rate)   
        inputAudios.append(audio)
input_features = processor(inputAudios, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device) 
                                  
        
# generate token ids
predicted_ids = model.generate(input_features)  
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)  
print(transcription)

with open(savePath+"transcribe.txt",'a',encoding="utf8") as f:
    for item in transcription:
        f.write(item+"\n")
    
    






