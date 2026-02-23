from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _latent_factors(n_entities: int) -> np.ndarray:
    # Simple 3D latent factors
    return np.random.randn(n_entities, 3).astype("float32")


def _generate_entities(out_dir: Path, n_entities: int, factors: np.ndarray) -> None:
    path = out_dir / "entities.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("entity_id,attributes,created_at\n")
        now = datetime(2025, 1, 1)
        for i in range(n_entities):
            eid = f"E{i:05d}"
            x, y, z = map(float, factors[i])
            target_reg = x + 0.5 * y - 0.2 * z
            target_bin = 1 if target_reg > 0 else 0
            target_rank = target_reg
            attrs = {
                "x": x,
                "y": y,
                "z": z,
                "target_regression": target_reg,
                "target_binary": target_bin,
                "target_ranking": target_rank,
            }
            created_at = now + timedelta(seconds=i)
            attrs_json = json.dumps(attrs, separators=(",", ":"))
            attrs_field = '"' + attrs_json.replace('"', '""') + '"'
            f.write(f"{eid},{attrs_field},{created_at.isoformat()}\n")


def _generate_events(out_dir: Path, n_events: int, n_entities: int, factors: np.ndarray) -> None:
    path = out_dir / "events.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("timestamp,entity_id,event_type,event_value,event_metadata\n")
        base_time = datetime(2025, 1, 1)
        for i in range(n_events):
            eidx = i % n_entities
            eid = f"E{eidx:05d}"
            x, y, z = map(float, factors[eidx])
            event_type = f"type_{(i % 5)}"
            value = x * math.sin(i / 10.0) + y * math.cos(i / 7.0) + 0.1 * z
            ts = base_time + timedelta(seconds=i)
            metadata = {"phase": int(i % 4)}
            metadata_json = json.dumps(metadata, separators=(",", ":"))
            metadata_field = '"' + metadata_json.replace('"', '""') + '"'
            row = f"{ts.isoformat()},{eid},{event_type},{value},{metadata_field}\n"
            f.write(row)


def _generate_interactions(out_dir: Path, n_interactions: int, n_entities: int, factors: np.ndarray) -> None:
    path = out_dir / "interactions.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("timestamp,src_entity_id,dst_entity_id,interaction_type,interaction_value,metadata\n")
        base_time = datetime(2025, 1, 1)
        for i in range(n_interactions):
            src_idx = i % n_entities
            dst_idx = (i * 7 + 3) % n_entities
            src_e = f"E{src_idx:05d}"
            dst_e = f"E{dst_idx:05d}"
            xs, ys, zs = map(float, factors[src_idx])
            xd, yd, zd = map(float, factors[dst_idx])
            itype = f"rel_{(i % 3)}"
            value = xs * xd + ys * yd + zs * zd
            ts = base_time + timedelta(seconds=i)
            metadata = {"strength": float(value)}
            metadata_json = json.dumps(metadata, separators=(",", ":"))
            metadata_field = '"' + metadata_json.replace('"', '""') + '"'
            row = f"{ts.isoformat()},{src_e},{dst_e},{itype},{value},{metadata_field}\n"
            f.write(row)


def _generate_image(latent: np.ndarray, size: Tuple[int, int] = (64, 64)) -> Image.Image:
    x, y, z = latent
    w, h = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for i in range(w):
        for j in range(h):
            r = int(127 + 120 * math.tanh(x + i / w))
            g = int(127 + 120 * math.tanh(y + j / h))
            b = int(127 + 120 * math.tanh(z + (i + j) / (w + h)))
            pixels[i, j] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return img


def _generate_audio(latent: np.ndarray, sr: int = 16000, seconds: float = 1.0) -> np.ndarray:
    x, y, z = latent
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype="float32")
    freq1 = 220.0 + 30.0 * x
    freq2 = 440.0 + 40.0 * y
    freq3 = 660.0 + 50.0 * z
    signal = (
        0.6 * np.sin(2 * math.pi * freq1 * t)
        + 0.3 * np.sin(2 * math.pi * freq2 * t)
        + 0.1 * np.sin(2 * math.pi * freq3 * t)
    ).astype("float32")
    return signal


def _generate_video_frames(
    latent: np.ndarray, frames: int = 8, size: Tuple[int, int] = (64, 64)
) -> List[Image.Image]:
    x, y, z = latent
    imgs: List[Image.Image] = []
    for k in range(frames):
        shift = k / max(1, frames - 1)
        img = _generate_image(np.array([x + shift * 0.1, y - shift * 0.1, z]), size=size)
        imgs.append(img)
    return imgs


def _write_video(path: Path, frames: List[Image.Image], fps: int = 8) -> None:
    import cv2
    import numpy as np

    if not frames:
        raise ValueError("No frames for video")
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for img in frames:
        arr = np.asarray(img.convert("RGB")).astype("uint8")
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()


def _generate_artifacts(out_dir: Path, n_artifacts: int, n_entities: int, factors: np.ndarray) -> None:
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "artifacts_manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("artifact_type,path,entity_id,timestamp,metadata\n")
        base_time = datetime(2025, 1, 1)
        for i in range(n_artifacts):
            ent_idx = i % n_entities
            eid = f"E{ent_idx:05d}"
            latent = factors[ent_idx]
            ts = base_time + timedelta(seconds=i)
            kind = ["image", "audio", "video"][i % 3]
            if kind == "image":
                img = _generate_image(latent)
                rel = f"img_{i:05d}.png"
                path = artifacts_dir / rel
                img.save(path)
            elif kind == "audio":
                audio = _generate_audio(latent)
                rel = f"aud_{i:05d}.wav"
                path = artifacts_dir / rel
                sf.write(str(path), audio, 16000)
            else:
                frames = _generate_video_frames(latent)
                rel = f"vid_{i:05d}.avi"
                path = artifacts_dir / rel
                _write_video(path, frames)
            meta = {"kind_index": int(i % 3)}
            meta_json = json.dumps(meta, separators=(",", ":"))
            meta_field = '"' + meta_json.replace('"', '""') + '"'
            row = f"{kind},{path},{eid},{ts.isoformat()},{meta_field}\n"
            f.write(row)


def generate_synthetic_dataset(
    out_dir: Path,
    n_entities: int,
    n_events: int,
    n_interactions: int,
    n_artifacts: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    factors = _latent_factors(n_entities)
    _generate_entities(out_dir, n_entities, factors)
    _generate_events(out_dir, n_events, n_entities, factors)
    _generate_interactions(out_dir, n_interactions, n_entities, factors)
    _generate_artifacts(out_dir, n_artifacts, n_entities, factors)
