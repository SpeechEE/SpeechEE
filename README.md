# SpeechEE

## SpeechEE: A Novel Benchmark for Speech Event Extraction

## ACMMM 2024 Paper


## Requirements

General

- Python (verified on 3.9)
- CUDA (verified on 11.8)

Python Packages

- refer to requirements.txt

```bash
conda create -n envs python=3.9
conda activate envs
pip install -r requirements.txt
```


## Datasets 
Please download the audio data from google drive links.

```
datasets
|
├── sentence-level
│   ├── PHEE
│   │   ├── train
│   │   │   ├── train-0.wav
│   │   │   ├── ...
│   │   ├── dev
│   │   │   ├── dev-0.wav
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── test-0.wav
│   │   │   ├── ...
|
├── document-level ...
|
├── dialogue-level ...


```


## Running SpeechEE

The command for training model is as follows:

```bash
bash run.bash -d 0 -f tree -m t5-large --label_smoothing 0 -l 5e-5 --lr_scheduler linear --warmup_steps 2000 -b 16 -i PHEE
```

- `-d` refers to the GPU device id.
- `-m t5-large` refers to using T5-large as the textual decoder.
- `-i` refers to the dataset.
