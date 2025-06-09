#!/usr/bin/env python
"""Execute model training steps of the ML pipeline."""
from src.pipelines.ml import step_7_0_train_models, step_7_5_ensemble


def main() -> None:
    """Run training and ensemble steps."""
    step_7_0_train_models.main()
    step_7_5_ensemble.main()


if __name__ == "__main__":
    main()
