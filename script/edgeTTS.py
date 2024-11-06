import asyncio
import edge_tts
import os
import json
import random
import time
import torch
import jsonlines

datasetDir = "RAMS"
datatype = "test"

path = os.path.join("data", datasetDir, datatype+".json")
sentence = []

with open(path, 'r',encoding='utf8') as f:
    data = f.readlines()

for item in data:
    line =json.loads(item)
    sentence.append(line["text"])

zh_voice_list = ['zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural', 'zh-CN-YunjianNeural', 'zh-CN-YunxiNeural',
                 'zh-CN-YunxiaNeural', 'zh-CN-YunyangNeural']

en_voice_list = ['en-US-AriaNeural','en-US-ChristopherNeural','en-US-EricNeural','en-US-GuyNeural','en-US-JennyNeural','en-US-MichelleNeural','en-US-RogerNeural',]



async def _main(text, voice, path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)
    print(path)


if __name__ == "__main__":


   
    for index, sten in enumerate(sentence):
        VOICE = zh_voice_list[random.randint(0, len(zh_voice_list)) - 1]
        print(index)
        print(sten)
        print(VOICE)
        na = datatype + "-" + str(index) + ".wav"
        audioPath = os.path.join(os.getcwd(), "output",datasetDir,datatype)
        audioPath = os.path.join(audioPath, na)
        print(audioPath)
        asyncio.run(_main(sten, VOICE, audioPath))
       


