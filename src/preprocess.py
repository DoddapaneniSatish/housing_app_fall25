from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]


DEFAULT_FEATURE_SPEC = FeatureSpec(
    numeric_cols=[
        "tenure",
        "monthly_charges",
        "total_charges",
        "senior_citizen",
    ],
    categorical_cols=[
        "gender",
        "contract_type",
        "payment_method",
        "internet_service",
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
        "multiple_lines",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ],
)


def build_preprocessor(
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> ColumnTransformer:
    num = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    cat = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num, spec.numeric_cols),
            ("cat", cat, spec.categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(
    use_pca: bool,
    pca_n_components: Optional[int] = None,
    spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
) -> Pipeline:
    steps = [
        ("preprocess", build_preprocessor(spec)),
    ]

    if use_pca:
        if pca_n_components is None:
            steps.append(("pca", PCA(n_components=0.95, random_state=42)))
        else:
            steps.append(("pca", PCA(n_components=pca_n_components, random_state=42)))

    return Pipeline(steps=steps)
