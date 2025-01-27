import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, audio=None):
        """
        qpos:   (B, 14)
        image:  (B, num_cameras, C=3, H, W)
        actions: (B, seq, 14) or None
        is_pad: (B, seq) boolean mask or None
        audio:  (B, 1, Freq, Time) or None

        Returns:
          - During training: a dict with loss terms
          - During inference: predicted actions (a_hat)
        """
        # If you have environment state, pass it here
        env_state = None

        # Example image normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        # If you have an audio transform, you can apply it here:
        # e.g.: audio = audio_transform(audio)

        if actions is not None:  # Training mode
            # Truncate actions and is_pad to model.num_queries if needed
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Pass audio to the model by specifying `audio=audio`
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, 
                image, 
                env_state, 
                actions=actions, 
                is_pad=is_pad,
                audio=audio
            )

            # KL divergence
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            # Example L1 loss
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            # Apply mask
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]  # total_kld is shape (1,) 
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict

        else:  # Inference mode
            # No actions => we sample from the prior
            a_hat, _, (_, _) = self.model(
                qpos, 
                image, 
                env_state,
                audio=audio
            )
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        """
        CNNMLPPolicy doesn't currently support audio, but you could add it similarly
        if your CNNMLP model were expanded to handle audio.
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    Compute KL divergence between N(mu, logvar.exp()) and N(0, 1).
    Returns total KLD, dimension-wise KLD, mean KLD.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld