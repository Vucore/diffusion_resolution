"""
functions/losses.py
-------------------
Noise estimation loss cho DDPM/DDIM — giữ nguyên logic gốc.

Công thức:
  x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
  Loss = ||ε - model(x_t, t)||²
"""
import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          keepdim=False):
    """
    Args:
        model: UNet model
        x0:    ảnh gốc (sạch), shape (N, C, H, W), đã rescale về [-1, 1]
        t:     timesteps ngẫu nhiên, shape (N,)
        e:     noise target ~ N(0,I), shape (N, C, H, W)
        b:     beta schedule, shape (T,)
        keepdim: nếu True, trả về loss per-sample thay vì trung bình

    Returns:
        scalar loss (mean) hoặc vector per-sample loss nếu keepdim=True
    """
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
