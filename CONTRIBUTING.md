# Contributing

Small, reproducible changes are easiest to review. Please do not attach participant data to issues or pull requests. Use synthetic DICOM/table examples whenever possible.

For development:

```bash
python -m pip install -e ".[dev]"
ruff check src tests
pytest
```

A pull request should explain the behavior being changed, include or update a test when practical, and avoid changing BIDS naming rules without a clear reason. Field-map behavior should stay conservative unless the source metadata are sufficient to make the BIDS role unambiguous.
