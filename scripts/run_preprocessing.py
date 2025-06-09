#!/usr/bin/env python
"""Execute preprocessing steps of the ML pipeline."""
from src.pipelines.ml import (
    step_1_merge_excels,
    step_2_generate_categories,
    step_3_clean_columns,
    step_4_transform_features,
    step_5_remove_relations,
    step_6_fpi_selection,
)


def main() -> None:
    """Run all preprocessing steps sequentially."""
    step_1_merge_excels.main()
    step_2_generate_categories.main()
    step_3_clean_columns.main()
    step_4_transform_features.main()
    step_5_remove_relations.main()
    step_6_fpi_selection.main()


if __name__ == "__main__":
    main()
