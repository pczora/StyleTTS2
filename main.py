from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from io import BytesIO
from flask import Flask, make_response, request
import wave
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import phonemizer
import nltk
from text_utils import TextCleaner
from utils import *
from models import *
from nltk.tokenize import word_tokenize
import librosa
import torchaudio
import yaml
import numpy as np
import random
import torch

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


random.seed(0)


np.random.seed(0)

textclenaer = TextCleaner()

nltk.download("punkt")

# TODO: Workaround for macOS, remove later
# _ESPEAK_LIBRARY = '/opt/homebrew/lib/libespeak-ng.dylib'
# EspeakWrapper.set_library(_ESPEAK_LIBRARY)


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(
        0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', preserve_punctuation=True, with_stress=True)

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model

BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load(
    "Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict

            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]


sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(
        sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
    clamp=False
)


def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,  # reference from the same speaker as the embedding
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                            F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    # weird pulse at the end of the model, need to be fixed later
    return out.squeeze().cpu().numpy()[..., :-50]


reference_dicts = {'1': "reference_audio/696_92939_000016_000006.wav",
                   '2': "reference_audio/1789_142896_000022_000005.wav", '3': "reference_audio/anger.wav",
                   '4': "reference_audio/sleepy.wav", '5': "reference_audio/amused.wav",
                   '6': "reference_audio/disgusted.wav"}

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    noise = torch.randn(1, 1, 256).to(device)
    text = request.json['text']
    voice_id = request.json['voice_id']
    print(voice_id)
    ref_s = compute_style(reference_dicts[voice_id])
    alpha = request.json.get('alpha', 0.3)
    print('alpha', alpha)
    beta = request.json.get('beta', 0.7)
    diffusion_steps = request.json.get('diffusion_steps', 5)
    embedding_scale = request.json.get('embedding_scale', 1)

    wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale)

    buf = BytesIO()
    with wave.open(buf, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes((wav * 32767).astype(np.int16).tobytes())
    response = make_response(buf.getvalue())
    buf.close()
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
