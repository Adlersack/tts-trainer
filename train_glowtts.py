import os
import re
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

root_path = "/tts-trainer/characters/"
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
             
             #print(f"Original: {colmn[1]} | Formatted: {text}")
             
             item_list.append({
                 "text": text,
                 "audio_file": audio_file,
                 "speaker_name": speaker,
                 "root_path": name_folder_path
             })
    
    return item_list

dataset_config = BaseDatasetConfig(
    formatter=None,
    meta_file_train="metadata.txt",
    language="en-us",
    path=name_folder_path
)

config = GlowTTSConfig(
    run_name="glowtts_custom_formatter",
    batch_size=32,
    eval_batch_size=64,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(name_folder_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    save_best_after=100,
    save_checkpoints=True,
    save_all_best=True,
    output_path="./output_training",
    mixed_precision=True,
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

audio_processor = AudioProcessor.init_from_config(config)
audio_processor.resample = True

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=config.eval_split_size,
    eval_split_max_size=config.eval_split_max_size,
    formatter=formatter
)

model = GlowTTS(config, audio_processor, tokenizer, speaker_manager=None)

print(f"Train Samples: {len(train_samples)}")
print(f"Eval Samples: {len(eval_samples)}")

trainer = Trainer(
    TrainerArgs(),
    config,
    output_path="tts-trainer/characters/output/output_training",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

if __name__ == '__main__':
    trainer.fit()