import torch
import whisper


def transcribe(audio_file: str) -> str:
    if torch.cuda.is_available():
        model = whisper.load_model("base").cuda()
    else:
        model = whisper.load_model("base")

    result = model.transcribe(
        audio_file, language="zh", initial_prompt="请转换成简体中文"
    )

    return result["text"].strip() if result["text"] else ""
