# Kairos - AI/ML Cost Intelligence Layer

Real-time GPU cost tracking for Jupyter notebooks. Know exactly how much your ML experiments cost.

## Quick Start

```bash
pip install kairos
```

```python
from kairos import KairosTracker

# Initialize tracker
tracker = KairosTracker()

# View current status
tracker.status()

# Track a cell execution
with tracker.track_cell(tags=["training"]):
    model.fit(X, y)

# Or log manually
tracker.log_cell(duration=120, tags=["inference"])

# View session summary
tracker.summary()
```

## Features

- **Real-time Cost Tracking**: See costs accumulate as you work
- **GPU Monitoring**: Automatic GPU detection and utilization tracking
- **Cloud Pricing**: Built-in pricing for AWS, GCP, and Azure instances
- **Rich Jupyter Output**: Beautiful HTML displays in notebooks
- **Cell-level Attribution**: Know exactly which cells cost the most
- **Optimization Recommendations**: Get actionable suggestions to reduce costs

## Cloud Provider Configuration

```python
# AWS
tracker = KairosTracker.for_aws("p4d.24xlarge")

# GCP
tracker = KairosTracker.for_gcp("a2-highgpu-8g")

# Azure
tracker = KairosTracker.for_azure("NC24ads_A100_v4")

# Custom rate
tracker = KairosTracker.with_rate(5.00)  # $5/hour
```

## Environment Variables

```bash
export KAIROS_CLOUD_PROVIDER=aws
export KAIROS_INSTANCE_TYPE=p4d.24xlarge
export KAIROS_HOURLY_RATE=32.77
export KAIROS_ALERT_THRESHOLD=100.00
```

## GPU Support

For GPU monitoring, install with:

```bash
pip install kairos[gpu]
```

This includes `pynvml` for NVIDIA GPU monitoring.

## License

MIT License - see LICENSE for details.

---

**[usekairos.ai](https://usekairos.ai)** - Eliminate AI/ML infrastructure waste
