import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def temp_path():
    """Create a temporary path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_images():
    """Create a batch of sample images."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_prediction_maps():
    """Create sample prediction maps."""
    return torch.sigmoid(torch.randn(4, 1, 224, 224))


@pytest.fixture
def sample_target_maps():
    """Create sample target maps."""
    return torch.randint(0, 2, (4, 1, 224, 224)).float()


@pytest.fixture
def sample_polygons():
    """Create sample polygon data."""
    return [
        np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32),
        np.array([[60, 10], [100, 10], [100, 30], [60, 30]], dtype=np.float32)
    ]


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        'encoder': {},
        'decoder': {},
        'head': {},
        'loss': {},
        'optimizer': {},
    })
    return config


# Enhanced fixtures for comprehensive testing
@pytest.fixture(scope="session")
def test_data_dir():
    """Root directory for test data."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image():
    """Sample image tensor for testing (enhanced version)."""
    return torch.randn(3, 512, 512)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    config_path = Path(__file__).parent / "fixtures" / "sample_config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf
        return OmegaConf.load(config_path)
    else:
        # Return comprehensive config if fixture file doesn't exist
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "model": {
                "encoder": {"backbone": "resnet18", "pretrained": False},
                "decoder": {"type": "unet"},
                "head": {"type": "db_head"},
                "loss": {"type": "db_loss"}
            },
            "data": {
                "batch_size": 2,
                "image_size": [256, 256]
            },
            "training": {
                "max_epochs": 1,
                "learning_rate": 0.001
            }
        })


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 256, 256),
                "polygons": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "text": "SAMPLE"
            }

    return MockDataset()


@pytest.fixture
def sample_batch():
    """Sample batch for testing."""
    return {
        "images": torch.randn(2, 3, 256, 256),
        "polygons": [
            [[10, 10, 50, 10, 50, 30, 10, 30]],
            [[20, 20, 60, 20, 60, 40, 20, 40]]
        ],
        "texts": ["HELLO", "WORLD"]
    }


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing."""
    return {
        "prob_maps": torch.sigmoid(torch.randn(2, 1, 256, 256)),
        "threshold_maps": torch.sigmoid(torch.randn(2, 1, 256, 256))
    }


@pytest.fixture
def sample_targets():
    """Sample ground truth targets for testing."""
    return torch.randint(0, 2, (2, 1, 256, 256)).float()


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


# GPU fixtures (only if CUDA is available)
@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture
def device(gpu_available):
    """Get appropriate device for testing."""
    return torch.device("cuda" if gpu_available else "cpu")


# Configuration fixtures for different architectures
@pytest.fixture
def dbnet_config():
    """DBNet architecture configuration."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "architecture": "dbnet",
        "model": {
            "encoder": {"backbone": "resnet50", "pretrained": False},
            "decoder": {"type": "unet"},
            "head": {"type": "db_head"},
            "loss": {"type": "db_loss"}
        }
    })


@pytest.fixture
def east_config():
    """EAST architecture configuration."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "architecture": "east",
        "model": {
            "encoder": {"backbone": "resnet50", "pretrained": False},
            "decoder": {"type": "unet"},
            "head": {"type": "east_head"},
            "loss": {"type": "east_loss"}
        }
    })


# Mock components for testing
@pytest.fixture
def mock_encoder():
    """Mock encoder for testing."""
    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.randn(x.shape[0], 256, x.shape[2]//32, x.shape[3]//32)

        @property
        def output_channels(self):
            return 256

        @property
        def output_stride(self):
            return 32

    return MockEncoder()


@pytest.fixture
def mock_decoder():
    """Mock decoder for testing."""
    class MockDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.randn(x.shape[0], 64, x.shape[2]*2, x.shape[3]*2)

    return MockDecoder()


@pytest.fixture
def mock_head():
    """Mock head for testing."""
    class MockHead(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])

        def postprocess(self, predictions, **kwargs):
            return {"polygons": [], "scores": []}

    return MockHead()


@pytest.fixture
def mock_loss():
    """Mock loss for testing."""
    class MockLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pred, target):
            return torch.tensor(0.5, requires_grad=True)

        def get_loss_components(self):
            return {"total": torch.tensor(0.5)}

    return MockLoss()