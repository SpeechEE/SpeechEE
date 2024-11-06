import torch
import torch.nn as nn
from transformers import WhisperModel



class Conv1dSubsampler(nn.Module):


    def __init__(
        self,
        in_channels: int = 1024,
        mid_channels: int = 1024,
        out_channels: int = 1024,
        kernel_sizes: List[int] = (3,3),
        stride: int = 2
    ):
        super(Conv1dSubsampler, self).__init__()
        self.stride = stride
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )


    def forward(self, src_tokens):
      
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)

        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T

        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)

        return x

class SpeechEEModel(nn.Module):
    def __init__(self, whisper_encoder, shrinking, textual_decoder):
        super(SpeechEEModel, self).__init__()
        self.whisper_encoder = whisper_encoder
        self.shrinking = shrinking
        self.textual_decoder = textual_decoder

    def forward(self, input_ids, attention_mask):
       
        encoder_output = self.whisper_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        encoder_output = self.shrinking(encoder_output)

        decoder_output = self.textual_decoder(inputs_embeds=encoder_output.last_hidden_state)

        return decoder_output



def build_model(whisper_path,decoder_model):
    whisper_encoder=WhisperModel.from_pretrained(whisper_path).encoder
    textual_decoder = decoder_model
    shrinking_model = Conv1dSubsampler()
    model = SpeechEEModel(whisper_encoder, shrinking_model, textual_decoder)
    return model


