"""UCR/TSML time-series data for the E006 continual-learning benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "e006_cache"


@dataclass(frozen=True)
class UCRSpec:
    """Immutable provenance and shape contract for one UCR binary task."""

    name: str
    display_name: str
    domain: str
    url: str
    archive_sha256: str
    train_size: int
    test_size: int
    original_length: int
    class_labels: tuple[str, str]
    class_names: tuple[str, str]


@dataclass(frozen=True)
class TemporalTask:
    """One binary time-series task with the official train/test split."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    spec: UCRSpec


SPECS = (
    UCRSpec(
        name="ECG200",
        display_name="ECG200 heartbeat",
        domain="ECG",
        url="https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip",
        archive_sha256="755937ea37849346ea20033666b19d17345bb039971e8fa33458b16b62c09380",
        train_size=100,
        test_size=100,
        original_length=96,
        class_labels=("-1", "1"),
        # The archive description names the two clinical classes but does not document
        # which one is encoded as -1 versus +1, so do not invent that mapping here.
        class_names=("archive label -1", "archive label 1"),
    ),
    UCRSpec(
        name="GunPoint",
        display_name="GunPoint motion",
        domain="human activity",
        url="https://www.timeseriesclassification.com/aeon-toolkit/GunPoint.zip",
        archive_sha256="d7513cfe222418fabfdb5a6434ffb21ac3de4923e637971e9388ebc857816803",
        train_size=50,
        test_size=150,
        original_length=150,
        class_labels=("1", "2"),
        class_names=("gun draw", "point"),
    ),
    UCRSpec(
        name="Coffee",
        display_name="Coffee spectrum",
        domain="spectrograph",
        url="https://www.timeseriesclassification.com/aeon-toolkit/Coffee.zip",
        archive_sha256="bfba9d67f4a0f2041cb8ba88b60b11f10dab60b670e40b904ac47bd7dfdf4749",
        train_size=28,
        test_size=28,
        original_length=286,
        class_labels=("0", "1"),
        class_names=("archive label 0", "archive label 1"),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_bytes(
    spec: UCRSpec,
    cache_dir: Path = DEFAULT_CACHE,
    *,
    allow_download: bool = True,
) -> bytes:
    """Return a verified archive, downloading it once when allowed."""
    path = cache_dir / f"{spec.name}.zip"
    if path.exists():
        data = path.read_bytes()
    else:
        if not allow_download:
            raise FileNotFoundError(f"missing cached dataset archive: {path}")
        request = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "qiskit-hackathon-e006/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    actual = _sha256(data)
    if actual != spec.archive_sha256:
        raise ValueError(
            f"{spec.name} archive SHA256 mismatch: expected {spec.archive_sha256}, got {actual}"
        )
    return data


def _parse_ts(text: str, spec: UCRSpec, expected_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse the equal-length, univariate subset of the aeon ``.ts`` format."""
    in_data = False
    series: list[list[float]] = []
    raw_labels: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower() == "@data":
                in_data = True
            continue
        try:
            values_text, label = line.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(f"invalid {spec.name} .ts data row") from exc
        values = [float(value) for value in values_text.split(",")]
        if len(values) != spec.original_length:
            raise ValueError(
                f"{spec.name} expected series length {spec.original_length}, got {len(values)}"
            )
        series.append(values)
        raw_labels.append(label.strip())

    if not in_data or len(series) != expected_size:
        raise ValueError(
            f"{spec.name} expected {expected_size} rows, got {len(series)}"
        )
    unknown = set(raw_labels).difference(spec.class_labels)
    if unknown:
        raise ValueError(f"{spec.name} contains unexpected labels: {sorted(unknown)}")
    features = np.asarray(series, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError(f"{spec.name} contains non-finite values")
    label_map = {spec.class_labels[0]: -1, spec.class_labels[1]: 1}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=int)
    return features, labels


def _reduce_series(features: np.ndarray, n_steps: int) -> np.ndarray:
    """Z-normalize each complete series and reduce it with equal-width PAA bins."""
    if features.ndim != 2 or n_steps <= 1 or n_steps > features.shape[1]:
        raise ValueError("features must be 2D and 1 < n_steps <= original length")
    means = features.mean(axis=1, keepdims=True)
    scales = features.std(axis=1, keepdims=True)
    if np.any(scales <= 1e-12):
        raise ValueError("constant time series cannot be normalized")
    normalized = (features - means) / scales
    bins = np.array_split(np.arange(features.shape[1]), n_steps)
    reduced = np.stack([normalized[:, indices].mean(axis=1) for indices in bins], axis=1)
    return np.clip(reduced, -3.0, 3.0) * (np.pi / 3.0)


def _load_task(
    spec: UCRSpec,
    *,
    n_steps: int,
    cache_dir: Path,
    allow_download: bool,
) -> TemporalTask:
    archive = _archive_bytes(spec, cache_dir, allow_download=allow_download)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        train_text = zipped.read(f"{spec.name}_TRAIN.ts").decode("utf-8-sig")
        test_text = zipped.read(f"{spec.name}_TEST.ts").decode("utf-8-sig")
    X_train, y_train = _parse_ts(train_text, spec, spec.train_size)
    X_test, y_test = _parse_ts(test_text, spec, spec.test_size)
    return TemporalTask(
        name=spec.display_name,
        X_train=_reduce_series(X_train, n_steps),
        y_train=y_train,
        X_test=_reduce_series(X_test, n_steps),
        y_test=y_test,
        spec=spec,
    )


def load_temporal_tasks(
    n_steps: int = 12,
    cache_dir: Path = DEFAULT_CACHE,
    *,
    allow_download: bool = True,
) -> tuple[TemporalTask, ...]:
    """Load ECG200 -> GunPoint -> Coffee using their official train/test splits."""
    cache_dir = Path(cache_dir)
    return tuple(
        _load_task(
            spec,
            n_steps=n_steps,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        for spec in SPECS
    )
