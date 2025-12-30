"""
Configuration management for Kairos cost tracking.

Handles GPU pricing, cloud provider configurations, and user preferences.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import json
import os


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class GPUType(Enum):
    """Common GPU types with identifiers."""
    # NVIDIA Consumer/Workstation
    RTX_4090 = "rtx_4090"
    RTX_3090 = "rtx_3090"
    RTX_3080 = "rtx_3080"

    # NVIDIA Data Center
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100 = "h100"
    V100 = "v100"
    T4 = "t4"
    L4 = "l4"
    A10G = "a10g"

    # Unknown/Generic
    UNKNOWN = "unknown"


@dataclass
class GPUPricing:
    """GPU pricing information per hour in USD."""
    gpu_type: GPUType
    provider: CloudProvider
    instance_type: str
    hourly_rate: float
    spot_rate: Optional[float] = None
    memory_gb: int = 0

    @property
    def spot_savings(self) -> float:
        """Calculate spot instance savings percentage."""
        if self.spot_rate and self.hourly_rate > 0:
            return (1 - self.spot_rate / self.hourly_rate) * 100
        return 0.0


# Default GPU pricing (USD per hour) - Updated December 2024
DEFAULT_GPU_PRICING: Dict[str, Dict[str, float]] = {
    # AWS Pricing
    "aws": {
        "p4d.24xlarge": 32.77,      # 8x A100 40GB
        "p4de.24xlarge": 40.97,     # 8x A100 80GB
        "p5.48xlarge": 98.32,       # 8x H100
        "p3.2xlarge": 3.06,         # 1x V100
        "p3.8xlarge": 12.24,        # 4x V100
        "p3.16xlarge": 24.48,       # 8x V100
        "g4dn.xlarge": 0.526,       # 1x T4
        "g4dn.12xlarge": 3.912,     # 4x T4
        "g5.xlarge": 1.006,         # 1x A10G
        "g5.12xlarge": 5.672,       # 4x A10G
        "g5.48xlarge": 16.288,      # 8x A10G
        "g6.xlarge": 0.8048,        # 1x L4
        "inf2.xlarge": 0.7582,      # AWS Inferentia2
    },
    # GCP Pricing
    "gcp": {
        "a2-highgpu-1g": 3.67,      # 1x A100 40GB
        "a2-highgpu-8g": 29.39,     # 8x A100 40GB
        "a2-ultragpu-1g": 5.00,     # 1x A100 80GB
        "a2-ultragpu-8g": 40.00,    # 8x A100 80GB
        "a3-highgpu-8g": 101.22,    # 8x H100
        "n1-standard-4-t4": 0.35,   # 1x T4
        "g2-standard-4": 0.83,      # 1x L4
    },
    # Azure Pricing
    "azure": {
        "NC6s_v3": 3.06,            # 1x V100
        "NC24s_v3": 12.24,          # 4x V100
        "NC24ads_A100_v4": 3.67,    # 1x A100 80GB
        "NC96ads_A100_v4": 14.69,   # 4x A100 80GB
        "ND96asr_v4": 27.20,        # 8x A100 40GB
        "ND96amsr_A100_v4": 32.77,  # 8x A100 80GB
        "NC4as_T4_v3": 0.526,       # 1x T4
    },
    # Local/Fallback pricing (estimated power cost + depreciation)
    "local": {
        "rtx_4090": 0.50,           # ~450W, amortized
        "rtx_3090": 0.40,           # ~350W, amortized
        "rtx_3080": 0.30,           # ~320W, amortized
        "a100": 1.50,               # Estimated local cost
        "v100": 0.80,               # Estimated local cost
        "t4": 0.25,                 # Estimated local cost
        "unknown": 0.35,            # Default fallback
    },
}

# Spot instance discount multipliers (approximate)
SPOT_DISCOUNT_RATES: Dict[str, float] = {
    "aws": 0.30,    # ~70% savings
    "gcp": 0.35,    # ~65% savings
    "azure": 0.40,  # ~60% savings
}


@dataclass
class KairosConfig:
    """
    Configuration for Kairos cost tracking.

    Attributes:
        cloud_provider: The cloud provider being used
        instance_type: The specific instance type (e.g., 'p4d.24xlarge')
        custom_hourly_rate: Override the default pricing with a custom rate
        auto_detect_gpu: Whether to automatically detect GPU type
        currency: Currency for display (default: USD)
        alert_threshold_usd: Alert when costs exceed this threshold
        auto_pause_idle_minutes: Auto-pause after this many idle minutes (0 = disabled)
        enable_html_output: Whether to render rich HTML in Jupyter
    """
    cloud_provider: CloudProvider = CloudProvider.LOCAL
    instance_type: Optional[str] = None
    custom_hourly_rate: Optional[float] = None
    auto_detect_gpu: bool = True
    currency: str = "USD"
    alert_threshold_usd: Optional[float] = None
    auto_pause_idle_minutes: int = 0
    enable_html_output: bool = True

    # Internal state
    _detected_gpu_type: Optional[GPUType] = field(default=None, repr=False)

    def get_hourly_rate(self) -> float:
        """
        Get the hourly rate for the current configuration.

        Returns:
            Hourly cost in USD
        """
        # Custom rate takes precedence
        if self.custom_hourly_rate is not None:
            return self.custom_hourly_rate

        provider_key = self.cloud_provider.value
        pricing = DEFAULT_GPU_PRICING.get(provider_key, {})

        # Try instance type first
        if self.instance_type and self.instance_type in pricing:
            return pricing[self.instance_type]

        # Fall back to detected GPU type for local
        if self._detected_gpu_type and provider_key == "local":
            gpu_key = self._detected_gpu_type.value
            if gpu_key in pricing:
                return pricing[gpu_key]

        # Ultimate fallback
        return DEFAULT_GPU_PRICING["local"].get("unknown", 0.35)

    def get_spot_rate(self) -> Optional[float]:
        """Get the estimated spot instance rate if available."""
        if self.cloud_provider == CloudProvider.LOCAL:
            return None

        hourly = self.get_hourly_rate()
        discount = SPOT_DISCOUNT_RATES.get(self.cloud_provider.value, 0.5)
        return hourly * discount

    @classmethod
    def from_env(cls) -> "KairosConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            KAIROS_CLOUD_PROVIDER: aws, gcp, azure, or local
            KAIROS_INSTANCE_TYPE: Instance type string
            KAIROS_HOURLY_RATE: Custom hourly rate override
            KAIROS_ALERT_THRESHOLD: Cost alert threshold in USD
        """
        config = cls()

        provider = os.environ.get("KAIROS_CLOUD_PROVIDER", "").lower()
        if provider in [p.value for p in CloudProvider]:
            config.cloud_provider = CloudProvider(provider)

        instance = os.environ.get("KAIROS_INSTANCE_TYPE")
        if instance:
            config.instance_type = instance

        rate = os.environ.get("KAIROS_HOURLY_RATE")
        if rate:
            try:
                config.custom_hourly_rate = float(rate)
            except ValueError:
                pass

        threshold = os.environ.get("KAIROS_ALERT_THRESHOLD")
        if threshold:
            try:
                config.alert_threshold_usd = float(threshold)
            except ValueError:
                pass

        return config

    @classmethod
    def for_aws(cls, instance_type: str) -> "KairosConfig":
        """Create configuration for AWS instance."""
        return cls(
            cloud_provider=CloudProvider.AWS,
            instance_type=instance_type,
        )

    @classmethod
    def for_gcp(cls, instance_type: str) -> "KairosConfig":
        """Create configuration for GCP instance."""
        return cls(
            cloud_provider=CloudProvider.GCP,
            instance_type=instance_type,
        )

    @classmethod
    def for_azure(cls, instance_type: str) -> "KairosConfig":
        """Create configuration for Azure instance."""
        return cls(
            cloud_provider=CloudProvider.AZURE,
            instance_type=instance_type,
        )

    def to_dict(self) -> Dict:
        """Serialize configuration to dictionary."""
        return {
            "cloud_provider": self.cloud_provider.value,
            "instance_type": self.instance_type,
            "custom_hourly_rate": self.custom_hourly_rate,
            "auto_detect_gpu": self.auto_detect_gpu,
            "currency": self.currency,
            "alert_threshold_usd": self.alert_threshold_usd,
            "auto_pause_idle_minutes": self.auto_pause_idle_minutes,
            "enable_html_output": self.enable_html_output,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KairosConfig":
        """Deserialize configuration from dictionary."""
        config = cls()

        if "cloud_provider" in data:
            config.cloud_provider = CloudProvider(data["cloud_provider"])
        if "instance_type" in data:
            config.instance_type = data["instance_type"]
        if "custom_hourly_rate" in data:
            config.custom_hourly_rate = data["custom_hourly_rate"]
        if "auto_detect_gpu" in data:
            config.auto_detect_gpu = data["auto_detect_gpu"]
        if "currency" in data:
            config.currency = data["currency"]
        if "alert_threshold_usd" in data:
            config.alert_threshold_usd = data["alert_threshold_usd"]
        if "auto_pause_idle_minutes" in data:
            config.auto_pause_idle_minutes = data["auto_pause_idle_minutes"]
        if "enable_html_output" in data:
            config.enable_html_output = data["enable_html_output"]

        return config
