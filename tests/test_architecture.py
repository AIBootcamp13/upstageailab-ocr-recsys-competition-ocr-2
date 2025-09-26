from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ocr.models.architecture import OCRModel


class TestOCRModel:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "encoder": {},
                "decoder": {},
                "head": {},
                "loss": {},
                "optimizer": {},
            }
        )
        return config

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224

    @patch("ocr.models.architecture.get_encoder_by_cfg")
    @patch("ocr.models.architecture.get_decoder_by_cfg")
    @patch("ocr.models.architecture.get_head_by_cfg")
    @patch("ocr.models.architecture.get_loss_by_cfg")
    def test_model_initialization(self, mock_loss, mock_head, mock_decoder, mock_encoder, mock_config):
        """Test that OCRModel initializes correctly with mocked components."""
        # Setup mocks
        mock_encoder.return_value = Mock()
        mock_decoder.return_value = Mock()
        mock_head.return_value = Mock()
        mock_loss.return_value = Mock()

        # Initialize model
        model = OCRModel(mock_config)

        # Verify components are created
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "head")
        assert hasattr(model, "loss")
        assert model.cfg == mock_config

        # Verify factory functions were called
        mock_encoder.assert_called_once_with(mock_config.encoder)
        mock_decoder.assert_called_once_with(mock_config.decoder)
        mock_head.assert_called_once_with(mock_config.head)
        mock_loss.assert_called_once_with(mock_config.loss)

    def test_forward_pass_with_loss(self, mock_config, sample_input):
        """Test forward pass that includes loss calculation."""
        with (
            patch("ocr.models.architecture.get_encoder_by_cfg") as mock_enc,
            patch("ocr.models.architecture.get_decoder_by_cfg") as mock_dec,
            patch("ocr.models.architecture.get_head_by_cfg") as mock_head,
            patch("ocr.models.architecture.get_loss_by_cfg") as mock_loss_func,
        ):
            # Setup component mocks
            mock_encoder = Mock()
            mock_decoder = Mock()
            mock_head_comp = Mock()
            mock_loss_comp = Mock()

            mock_enc.return_value = mock_encoder
            mock_dec.return_value = mock_decoder
            mock_head.return_value = mock_head_comp
            mock_loss_func.return_value = mock_loss_comp

            # Setup return values
            encoded_features = torch.randn(2, 64, 56, 56)
            decoded_features = torch.randn(2, 32, 112, 112)
            pred = {"maps": torch.randn(2, 1, 224, 224)}
            loss = 0.5
            loss_dict = {"total_loss": loss}

            mock_encoder.return_value = encoded_features
            mock_decoder.return_value = decoded_features
            mock_head_comp.return_value = pred
            mock_loss_comp.return_value = (loss, loss_dict)

            # Initialize and run model
            model = OCRModel(mock_config)
            result = model(sample_input, return_loss=True, gt_maps=torch.randn(2, 1, 224, 224))

            # Verify result structure
            assert "maps" in result
            assert "loss" in result
            assert "loss_dict" in result
            assert result["loss"] == loss
            assert result["loss_dict"] == loss_dict

    def test_forward_pass_without_loss(self, mock_config, sample_input):
        """Test forward pass without loss calculation."""
        with (
            patch("ocr.models.architecture.get_encoder_by_cfg") as mock_enc,
            patch("ocr.models.architecture.get_decoder_by_cfg") as mock_dec,
            patch("ocr.models.architecture.get_head_by_cfg") as mock_head,
            patch("ocr.models.architecture.get_loss_by_cfg") as mock_loss_func,
        ):
            # Setup component mocks
            mock_encoder = Mock()
            mock_decoder = Mock()
            mock_head_comp = Mock()

            mock_enc.return_value = mock_encoder
            mock_dec.return_value = mock_decoder
            mock_head.return_value = mock_head_comp
            mock_loss_func.return_value = Mock()

            # Setup return values
            encoded_features = torch.randn(2, 64, 56, 56)
            decoded_features = torch.randn(2, 32, 112, 112)
            pred = {"maps": torch.randn(2, 1, 224, 224)}

            mock_encoder.return_value = encoded_features
            mock_decoder.return_value = decoded_features
            mock_head_comp.return_value = pred

            # Initialize and run model
            model = OCRModel(mock_config)
            result = model(sample_input, return_loss=False)

            # Verify result structure
            assert "maps" in result
            assert "loss" not in result
            assert "loss_dict" not in result

    @patch("ocr.models.architecture.instantiate")
    def test_get_optimizers_without_scheduler(self, mock_instantiate, mock_config):
        """Test optimizer creation without scheduler."""
        with (
            patch("ocr.models.architecture.get_encoder_by_cfg"),
            patch("ocr.models.architecture.get_decoder_by_cfg"),
            patch("ocr.models.architecture.get_head_by_cfg"),
            patch("ocr.models.architecture.get_loss_by_cfg"),
        ):
            mock_optimizer = Mock()
            mock_instantiate.return_value = mock_optimizer

            model = OCRModel(mock_config)
            optimizers, schedulers = model.get_optimizers()

            assert len(optimizers) == 1
            assert len(schedulers) == 0
            assert optimizers[0] == mock_optimizer
            assert schedulers == []
            mock_instantiate.assert_called_once()
            args, kwargs = mock_instantiate.call_args
            assert args[0] == mock_config.optimizer
            assert "params" in kwargs

    @patch("ocr.models.architecture.instantiate")
    def test_get_optimizers_with_scheduler(self, mock_instantiate, mock_config):
        """Test optimizer and scheduler creation."""
        with (
            patch("ocr.models.architecture.get_encoder_by_cfg"),
            patch("ocr.models.architecture.get_decoder_by_cfg"),
            patch("ocr.models.architecture.get_head_by_cfg"),
            patch("ocr.models.architecture.get_loss_by_cfg"),
        ):
            mock_optimizer = Mock()
            mock_scheduler = Mock()
            mock_instantiate.side_effect = [mock_optimizer, mock_scheduler]

            # Add scheduler to config
            from omegaconf import OmegaConf

            OmegaConf.set_struct(mock_config, False)
            mock_config.scheduler = {}

            model = OCRModel(mock_config)
            optimizers, schedulers = model.get_optimizers()

            assert len(optimizers) == 1
            assert len(schedulers) == 1
            assert optimizers[0] == mock_optimizer
            assert schedulers[0] == mock_scheduler

    def test_get_polygons_from_maps(self, mock_config):
        """Test polygon extraction from prediction maps."""
        with (
            patch("ocr.models.architecture.get_encoder_by_cfg"),
            patch("ocr.models.architecture.get_decoder_by_cfg"),
            patch("ocr.models.architecture.get_head_by_cfg") as mock_head,
            patch("ocr.models.architecture.get_loss_by_cfg"),
        ):
            mock_head_comp = Mock()
            mock_head.return_value = mock_head_comp

            gt_maps = torch.randn(2, 1, 224, 224)
            pred_maps = torch.randn(2, 1, 224, 224)
            expected_polygons = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

            mock_head_comp.get_polygons_from_maps.return_value = expected_polygons

            model = OCRModel(mock_config)
            result = model.get_polygons_from_maps(gt_maps, pred_maps)

            assert result == expected_polygons
            mock_head_comp.get_polygons_from_maps.assert_called_once_with(gt_maps, pred_maps)
