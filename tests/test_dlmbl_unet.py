import numpy as np
import pytest
import torch

import dlmbl_unet


class TestDown:
    def test_shape_checker(self) -> None:
        down2 = dlmbl_unet.Downsample(2)
        msg = "Your `check_valid` function is not right yet."
        assert down2.check_valid((8, 8)), msg
        assert not down2.check_valid((9, 9)), msg
        assert not down2.check_valid((9, 8)), msg
        assert not down2.check_valid((8, 9)), msg
        down2_3d = dlmbl_unet.Downsample(2, ndim=3)
        assert down2_3d.check_valid((8, 8, 8)), msg
        assert not down2_3d.check_valid((9, 9, 9)), msg
        assert not down2_3d.check_valid((9, 8, 8)), msg
        assert not down2_3d.check_valid((8, 9, 8)), msg
        assert not down2_3d.check_valid((8, 8, 9)), msg
        down3 = dlmbl_unet.Downsample(3)
        assert down3.check_valid((9, 9)), msg
        assert not down3.check_valid((8, 8)), msg
        assert not down3.check_valid((9, 8)), msg
        assert not down3.check_valid((8, 9)), msg
        down3_3d = dlmbl_unet.Downsample(3, ndim=3)
        assert down3_3d.check_valid((9, 9, 9)), msg
        assert not down3_3d.check_valid((8, 8, 8)), msg
        assert not down3_3d.check_valid((8, 9, 9)), msg
        assert not down3_3d.check_valid((9, 8, 9)), msg
        assert not down3_3d.check_valid((9, 9, 8)), msg

    def test_shape_checker_error(self) -> None:
        down2 = dlmbl_unet.Downsample(2)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 7, 8))
            down2(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 8, 7))
            down2(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((4, 7, 8))
            down2(x)
        down2_3d = dlmbl_unet.Downsample(2, ndim=3)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 7, 8, 8))
            down2_3d(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 8, 7, 8))
            down2_3d(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 4, 8, 8, 7))
            down2_3d(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 7, 8, 8))
            down2_3d(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 8, 7, 8))
            down2_3d(x)
        with pytest.raises(RuntimeError):
            x = torch.zeros((2, 8, 8, 7))
            down2_3d(x)

    def test_shape(self) -> None:
        tensor2 = torch.arange(16).reshape((1, 4, 4))
        down2 = dlmbl_unet.Downsample(2)
        expected = torch.Tensor([5, 7, 13, 15]).reshape((1, 2, 2))
        msg = "The output shape of your Downsample module is not correct."
        assert expected.shape == down2(tensor2).shape, msg
        msg = "The ouput shape of your Downsample module is correct, but the values are not."
        assert torch.equal(expected, down2(tensor2)), msg


class TestConvBlock:
    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_valid(self, batch_norm) -> None:
        shape = [20, 30]
        channels = 4
        out_channels = 5
        kernel_size = 7
        n_batch = 1

        tensor_in = torch.ones([n_batch, channels, *shape])
        conv = dlmbl_unet.ConvBlock(
            channels, out_channels, kernel_size, padding="valid", use_batch_norm=batch_norm
        )
        tensor_out = conv(tensor_in)

        shape_expected = list(np.array(shape) - 2 * (kernel_size - 1))
        shape_expected = [n_batch, out_channels, *shape_expected]
        msg = "Output shape for valid padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_same(self, batch_norm) -> None:
        shape = [16, 39]
        channels = 4
        out_channels = 5
        kernel_size = 7
        n_batch = 1

        tensor_in = torch.ones([n_batch, channels, *shape])
        conv = dlmbl_unet.ConvBlock(channels, out_channels, kernel_size, padding="same", use_batch_norm=batch_norm)
        tensor_out = conv(tensor_in)

        shape_expected = [n_batch, out_channels, *shape]
        msg = "Output shape for same padding is incorrect."
        assert tensor_out.shape == torch.Size(shape_expected), msg

    def test_relu(self) -> None:
        shape = [1, 1, 100, 100]
        tensor_in = torch.randn(shape) * 2

        conv = dlmbl_unet.ConvBlock(1, 50, 5, padding="same")
        tensor_out = conv(tensor_in)
        msg = "Your activation function is incorrect."
        assert torch.all(tensor_out >= 0), msg


class TestCropAndConcat:
    def test_crop(self) -> None:
        big_tensor = torch.ones((12, 14, 40, 50))
        small_tensor = torch.zeros((12, 5, 13, 18))
        ccmod = dlmbl_unet.CropAndConcat()
        out_tensor = ccmod(big_tensor, small_tensor)
        expected_tensor = torch.cat(
            [torch.ones(12, 14, 13, 18), torch.zeros(12, 5, 13, 18)], dim=1
        )
        msg = "Your CropAndConcat node does not give the expected output"
        assert torch.equal(out_tensor, expected_tensor), msg


class TestOutputConv:
    def test_channels(self) -> None:
        outconv = dlmbl_unet.OutputConv(3, 30, activation=torch.nn.Softshrink())
        tensor = torch.ones((3, 24, 17))
        tensor_out = outconv(tensor)
        msg = "The output shape of your output conv is not right."
        assert tensor_out.shape == torch.Size((30, 24, 17)), msg


class TestUNet:
    def test_fmaps(self) -> None:
        unet = dlmbl_unet.UNet(5, 1, 1, num_fmaps=17, fmap_inc_factor=4)
        msg = "The computation of number of feature maps in the encoder is incorrect"
        assert unet.compute_fmaps_encoder(3) == (272, 1088), msg
        msg = "The computation of number of feature maps in the decoder is incorrect"
        assert unet.compute_fmaps_decoder(3) == (5440, 1088), msg
        msg = "The computation of number of feature maps in the encoder is incorrect for level 0"
        assert unet.compute_fmaps_encoder(0) == (1, 17), msg
        msg = "The computation of number of feature maps in the decoder is incorrect for level 0"
        assert unet.compute_fmaps_decoder(0) == (85, 17), msg

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_valid_2d(self, batch_norm) -> None:
        unetvalid = dlmbl_unet.UNet(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid",
            use_batch_norm=batch_norm
        )
        msg = "The output shape of your UNet is incorrect for valid padding."
        assert unetvalid(torch.ones((2, 2, 536, 536))).shape == torch.Size(
            (2, 7, 108, 108)
        ), msg

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_valid_3d(self, batch_norm) -> None:
        unetvalid = dlmbl_unet.UNet(
            depth=3,
            in_channels=2,
            out_channels=1,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="valid",
            ndim=3,
            use_batch_norm=batch_norm
        )
        msg = "The output shape of your UNet is incorrect for valid padding in 3D."
        assert unetvalid(torch.ones((2, 2, 158, 158, 158))).shape == torch.Size(
            (2, 1, 18, 18, 18)
        ), msg

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_same_2d(self, batch_norm) -> None:
        unetsame = dlmbl_unet.UNet(
            depth=4,
            in_channels=2,
            out_channels=7,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="same",
            use_batch_norm=batch_norm
        )
        msg = "The output shape of your Unet is incorrect for same padding."
        assert unetsame(torch.ones((2, 2, 243, 243))).shape == torch.Size(
            (2, 7, 243, 243)
        ), msg

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_shape_same_3d(self, batch_norm) -> None:
        unetsame = dlmbl_unet.UNet(
            depth=3,
            in_channels=2,
            out_channels=1,
            num_fmaps=5,
            fmap_inc_factor=5,
            downsample_factor=3,
            kernel_size=5,
            padding="same",
            ndim=3,
            use_batch_norm=batch_norm
        )
        msg = "The output shape of your Unet is incorrect for same padding."
        assert unetsame(torch.ones((2, 2, 27, 27, 27))).shape == torch.Size(
            (2, 1, 27, 27, 27)
        ), msg


    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_has_batch_norm_2d(self, batch_norm):
        unet = dlmbl_unet.UNet(
            depth=3,
            in_channels=2,
            ndim=2,
            use_batch_norm=batch_norm
        )
        if batch_norm:
            module_types = [type(m) for m in unet.modules()]
            assert torch.nn.BatchNorm2d in module_types
        else:
            module_types = [type(m) for m in unet.modules()]
            assert torch.nn.BatchNorm2d not in module_types

    @pytest.mark.parametrize('batch_norm', [True, False])
    def test_has_batch_norm_3d(self, batch_norm):
        unet = dlmbl_unet.UNet(
            depth=3,
            in_channels=2,
            ndim=3,
            use_batch_norm=batch_norm
        )
        if batch_norm:
            module_types = [type(m) for m in unet.modules()]
            assert torch.nn.BatchNorm3d in module_types
        else:
            module_types = [type(m) for m in unet.modules()]
            assert torch.nn.BatchNorm3d not in module_types