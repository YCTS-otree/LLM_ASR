"""FP32 islands for numerically sensitive Paraformer operations under AMP."""
import torch


def float_output(module, inputs, output):
    return float_inputs(output)


def protect_residuals(model):
    # The first SANM block changes dimensions and creates a half residual under
    # AMP. Subsequent additions can exceed FP16 range before LayerNorm runs.
    # Promote residual branches while keeping expensive Linear GEMMs under AMP.
    for module in model.modules():
        if module.__class__.__name__ in ('EncoderLayerSANM', 'DecoderLayerSANM'):
            module.dropout.register_forward_hook(float_output)
            # SANM's FFN output projection can itself exceed 65504 before
            # the residual addition. Keep this projection in FP32 as well.
            feed_forward = getattr(module, 'feed_forward', None)
            if feed_forward is not None and hasattr(feed_forward, 'w_2'):
                feed_forward.w_2 = Float32Module(feed_forward.w_2)


def float_inputs(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(float_inputs(item) for item in value)
    if isinstance(value, list):
        return [float_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: float_inputs(item) for key, item in value.items()}
    return value


class Float32Module(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            return self.module(*float_inputs(args), **float_inputs(kwargs))
