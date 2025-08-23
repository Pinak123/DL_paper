import torch
from model import PatchDiscriminator, UNetGenerator


def test_discriminator_shapes():
    batch_size = 2
    in_channels = 3
    H, W = 256, 256

    D = PatchDiscriminator(in_channels=in_channels)
    x = torch.randn(batch_size, in_channels, H, W)
    y = torch.randn(batch_size, in_channels, H, W)
    with torch.no_grad():
        out = D(x, y)

    # With our configuration (strides: 2,2,2,1 + final 1), 256x256 → 30x30
    assert out.shape[0] == batch_size
    assert out.shape[1] == 1
    assert out.shape[2] == 30 and out.shape[3] == 30


def test_generator_shapes():
    batch_size = 2
    in_channels = 3
    out_channels = 3
    H, W = 256, 256

    G = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
    x = torch.randn(batch_size, in_channels, H, W)
    with torch.no_grad():
        y = G(x)

    assert y.shape == (batch_size, out_channels, H, W)


if __name__ == "__main__":
    test_discriminator_shapes()
    test_generator_shapes()
    print("All tests passed.")


