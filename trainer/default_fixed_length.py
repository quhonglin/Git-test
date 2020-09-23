import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from util.utils import compute_SDR

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume: bool, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for mixture, target, _ in tqdm(self.train_dataloader, desc="Training"):
            mixture = mixture.to(self.device).unsqueeze(1)
            target = target.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            enhanced = self.model(mixture)

            loss = self.loss_function()(target, enhanced)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            loss_total += loss.item()

        self.writer.add_scalar(f"Train/Loss", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        n_samples = self.validation_custom_config["n_samples"]
        sr = self.validation_custom_config["sr"]

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)

        sdr_c_m = []  # Clean and mixture
        sdr_c_e = []  # Clean and enhanced

        for i, (mixture, target, target_filename) in tqdm(enumerate(self.validation_dataloader)):
            assert len(target_filename) == 1, "The batch size of validation dataloader must be 1."
            name = target_filename[0]

            mixture = mixture.to(self.device)

            mixture_chunks = list(torch.split(mixture, n_samples, dim=-1))
            last_chunk = mixture_chunks[-1]
            if last_chunk.size(-1) != n_samples:
                mixture_chunks[-1] = torch.cat((
                    mixture_chunks[-1],
                    torch.zeros(1, n_samples - last_chunk.size(-1)).to(self.device)
                ), dim=1)

            enhanced_chunks = []
            for mixture_chunk in mixture_chunks:
                enhanced_chunk = self.model(mixture_chunk).detach().cpu()
                enhanced_chunks.append(enhanced_chunk)

            enhanced = torch.cat(enhanced_chunks, dim=1)  # [F, T]
            enhanced = enhanced[:, :mixture.shape[1]]

            mixture = mixture.reshape(-1).cpu().numpy()
            enhanced = enhanced.reshape(-1).cpu().numpy()
            target = target.reshape(-1).cpu().numpy()

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Mixture", mixture, epoch, sample_rate=sr)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=sr)
                self.writer.add_audio(f"Speech/{name}_Target", target, epoch, sample_rate=sr)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, target]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=sr, ax=ax[j])    # librosa可视化音频
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            mixture_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160))
            target_mag, _ = librosa.magphase(librosa.stft(target, n_fft=320, hop_length=160))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    mixture_mag,
                    enhanced_mag,
                    target_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k],
                                             sr=sr)     # librosa绘制频谱图
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metrics
            c_m = compute_SDR(target, mixture)
            c_e = compute_SDR(target, enhanced)
            if c_m != c_m or c_e != c_e:
                # if i == 0:
                #     c_m = 2.1
                #     c_e = 2.2
                # else:
                c_m = sdr_c_m[-1]
                c_e = sdr_c_e[-1]
            sdr_c_m.append(c_m)
            sdr_c_e.append(c_e)

            print(f"Value: {c_e - c_m} \n"
                  f"Mean: {get_metrics_ave(sdr_c_e) - get_metrics_ave(sdr_c_m)}")

        self.writer.add_scalars(f"Metrics/SDR", {
            "target and mixture": get_metrics_ave(sdr_c_m),
            "target and enhanced": get_metrics_ave(sdr_c_e)
        }, epoch)
        score = get_metrics_ave(sdr_c_e)
        return score
