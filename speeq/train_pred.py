import torch.nn as nn

from speeq.models import registry, templates
from speeq.config import ModelConfig, ASRDataConfig, TrainerConfig
from speeq.data.processors import OrderedProcessor, StochasticProcessor
from speeq.data.augmenters import FrequencyMasking, WhiteNoiseInjector
from speeq.data.processes import AudioLoader, FeatExtractor, FeatStacker, FrameContextualizer
from speeq.interfaces import IProcess
from speeq.trainers.trainers import launch_training_job


class TextStripper(IProcess):
    """Strip leading and trailing whitespace from text."""
    def run(self, text: str) -> str:
        return text.strip()


class TextLowering(IProcess):
    """Convert text to lowercase."""
    def run(self, text: str) -> str:
        return text.lower()


def main():
    # 1. Định nghĩa template cho Transformer Transducer
    template = templates.TransformerTransducerTemp(
        in_features=80,
        n_layers=18,
        n_dec_layers=2,
        d_model=512,
        ff_size=2048,
        h=8,
        joint_size=512,
        enc_left_size=10,
        dec_left_size=2,
        enc_right_size=0,
        dec_right_size=0,
        p_dropout=0.1,
        stride=1,
        kernel_size=1,
        masking_value=-1e15
    )

    # 2. Tạo ModelConfig và model
    model_cfg = ModelConfig(template=template)
    model = registry.get_model(model_config=model_cfg, n_classes=5)

    # 3. Tạo text_processor
    text_processor = OrderedProcessor([
        TextStripper(),
        TextLowering()
    ])

    # 4. Tạo speech_processor
    speech_processor = OrderedProcessor([
        AudioLoader(sample_rate=16000),
        FeatExtractor(feat_ext_name='melspec', feat_ext_args={})
    ])

    # 5. Cấu hình dữ liệu (ASRDataConfig)
    data_cfg = ASRDataConfig(
        training_path='path/to/train.csv',
        testing_path='path/to/test.csv',
        speech_processor=speech_processor,
        text_processor=text_processor,
        tokenizer_path='outdir/tokenizer.json',
        tokenizer_type='word_tokenizer',
        add_sos_token=True,
        add_eos_token=True,
    )

    # 6. Tạo TrainerConfig
    single_gpu_trainer_cfg = TrainerConfig(
        name='transducer',  # hoặc 'ctc', 'seq2seq' nếu dùng model khác
        batch_size=4,
        epochs=100,
        outdir='outdir',
        logdir='outdir/logs',
        log_steps_frequency=100,
        criterion='rnnt',   # nếu dùng transducer thì criterion nên là 'rnnt'
        optimizer='adam',
        optim_args={'lr': 0.0001},
        device='cuda'
    )

    # 7. Gọi hàm launch_training_job để bắt đầu train
    launch_training_job(
        trainer_config=single_gpu_trainer_cfg,
        data_config=data_cfg,
        model_config=model_cfg
    )


if __name__ == '__main__':
    main()
