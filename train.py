import os
import re
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

root_path = "./characters"
name = 'Ganyu'
name_folder_path = os.path.join(root_path, name)
name_audio_path = os.path.join(name_folder_path + '/audio')
metadata = 'metadata.txt'

def formatter(name_folder_path, metadata, **kwargs):
    regex = r'([+#"]+)|(<.*?>+)|{.*?}+'
    file = os.path.join(name_folder_path, metadata)
    item_list = []
    speaker = name
    
    with open(file, 'r', encoding='utf-8') as tf:
         for line in tf:
             colmn = line.split('|')
             audio_file = f"{name_audio_path}/{colmn[0]}.wav"
             text = re.sub(regex, "", colmn[1])
             
             item_list.append({
                 "text": text,
                 "audio_file": audio_file,
                 "speaker_name": speaker,
                 "root_path": name_folder_path
             })
    
    return item_list
    
    

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.txt",
    language="en-us",
    path=name_folder_path
)

audio_config = VitsAudioConfig(
    sample_rate=48000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)

character_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890",
    punctuations=" .,!?-",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)

config = VitsConfig(
    audio=audio_config,
    characters=character_config,
    run_name="vits_vctk",
    batch_size=4,
    eval_batch_size=2,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(name_folder_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    save_best_after=100,
    save_checkpoints=True,
    save_all_best=True,
    mixed_precision=True,
    max_text_len=150,
    output_path=name_folder_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences=[
        ["Hello, I hope you are doing well!"],
        ["It was nice seeing you again. I hope I didn't sound too robotic."],
        ["Farewell."]
    ]
)

audio_processor = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=formatter
)

model = Vits(config, audio_processor, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path=name_folder_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

if __name__ == '__main__':
    trainer.fit()