import os
import jsonlines
import random
import time
import scipy

import torch

datasetDir = "ace05-EN"
datatype = "dev"
path = os.path.join("/data/xxx/TTS","data", datasetDir, datatype+".json")
sentence = []
speaker = []

with open(path,'rb') as f:

    for item in jsonlines.Reader(f):
        sentence.append(item["sentence"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(len(sentence))

voicePath = "/root/.cache/huggingface/hub/models--ylacombe--bark-large/speaker_embeddings"
modelPath = "/root/.cache/huggingface/hub/models--suno--bark"
processor = AutoProcessor.from_pretrained(modelPath)
model = BarkModel.from_pretrained(modelPath).to(device)


en_voice_list = ["v2/en_speaker_9","v2/en_speaker_6","v2/en_speaker_5","v2/en_speaker_4","v2/en_speaker_3","v2/en_speaker_2","v2/en_speaker_9","v2/en_speaker_1","v2/en_speaker_0","v2/en_speaker_9","v2/en_speaker_9","v2/en_speaker_8","v2/en_speaker_7","v2/en_speaker_6","v2/en_speaker_6"]



processor = AutoProcessor.from_pretrained(modelPath)

T1 = time.time()
for index,sten in enumerate(sentence):
    voice_preset = en_voice_list[random.randint(0, len(en_voice_list)) - 1]
    print(voice_preset)
    inputs = processor(sten, voice_preset=voice_preset).to(device)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    na =datatype+"-"+str(index)+".wav"
    audioPath = "/data/xxx/TTS/output/ACE05-E/train"
    audioPath =os.path.join(audioPath,na)
    print(audioPath)
    scipy.io.wavfile.write(audioPath, rate=sample_rate, data=audio_array)



T2 = time.time()
print('time: %sminutes' % ((T2 - T1)/60))


