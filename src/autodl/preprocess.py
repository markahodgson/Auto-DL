from __future__ import annotations

from dataclasses import asdict, dataclass
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from autodl.config import AppConfig
from autodl.profiling import profile_dataframe


@dataclass
class PreprocessResult:
    transformed: pd.DataFrame
    metadata: dict


def _text_features(df: pd.DataFrame, text_cols: list[str], config: AppConfig) -> pd.DataFrame:
    if not text_cols or config.preprocess.text_backend == "none":
        return pd.DataFrame(index=df.index)

    feature_blocks: list[pd.DataFrame] = []
    backend = config.preprocess.text_backend

    for col in text_cols:
        text = df[col].fillna("").astype(str)
        if backend == "hashing":
            vec = HashingVectorizer(
                n_features=config.preprocess.text_hash_features,
                alternate_sign=False,
                norm="l2",
            )
            m = vec.transform(text)
            block = pd.DataFrame.sparse.from_spmatrix(m)
            block.columns = [f"{col}__hash_{i}" for i in range(block.shape[1])]
            feature_blocks.append(block)
        elif backend == "tfidf":
            vec = TfidfVectorizer(max_features=config.preprocess.max_text_vocab)
            m = vec.fit_transform(text)
            block = pd.DataFrame.sparse.from_spmatrix(m)
            block.columns = [f"{col}__tfidf_{i}" for i in range(block.shape[1])]
            feature_blocks.append(block)
        else:
            feature_blocks.append(pd.DataFrame({f"{col}__raw": text.values}, index=df.index))

    if feature_blocks:
        return pd.concat(feature_blocks, axis=1)
    return pd.DataFrame(index=df.index)


def preprocess_dataframe(df: pd.DataFrame, target: str, config: AppConfig) -> PreprocessResult:
    profile = profile_dataframe(df, target=target, passthrough_cols=config.preprocess.passthrough_columns)

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    for col in profile.numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in profile.categorical_cols:
        X[col] = X[col].astype("string").fillna("__MISSING__")

    for col in profile.text_cols:
        X[col] = X[col].astype("string").fillna("")

    if config.preprocess.normalize_numeric and profile.numeric_cols:
        scaler = StandardScaler()
        X[profile.numeric_cols] = scaler.fit_transform(X[profile.numeric_cols])

    cat_frame = pd.DataFrame(index=X.index)
    if config.preprocess.one_hot_categorical and profile.categorical_cols:
        cat_frame = pd.get_dummies(
            X[profile.categorical_cols],
            columns=profile.categorical_cols,
            dtype="int8",
            sparse=True,
        )

    numeric_frame = X[profile.numeric_cols] if profile.numeric_cols else pd.DataFrame(index=X.index)
    passthrough_frame = X[profile.passthrough_cols] if profile.passthrough_cols else pd.DataFrame(index=X.index)
    text_frame = _text_features(X, profile.text_cols, config)

    out = pd.concat([numeric_frame, cat_frame, text_frame, passthrough_frame], axis=1)
    out[target] = y.values

    metadata = {
        "profile": asdict(profile),
        "n_rows": int(out.shape[0]),
        "n_cols": int(out.shape[1]),
        "text_backend": config.preprocess.text_backend,
    }

    return PreprocessResult(transformed=out, metadata=metadata)
