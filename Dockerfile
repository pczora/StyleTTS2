FROM --platform=amd64 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
WORKDIR /styletts2
RUN apt-get update && apt-get install -y espeak-ng python3 python3-pip git curl unzip
RUN curl -LO https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip \
    && unzip reference_audio.zip

RUN mkdir -p Models/LibriTTS \
    && curl -LO -o Models/LibriTTS/config.yml https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml \
    && curl -LO -o Models/LibriTTS/epochs_2nd_00020.pth https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth

COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]