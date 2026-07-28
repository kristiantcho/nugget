from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch


def _metadata_path(parquet_path: Path) -> Path:
	return parquet_path.with_suffix(parquet_path.suffix + ".meta.json")


def _validate_signal_events(signal_events: Sequence[Dict]) -> List[str]:
	if len(signal_events) == 0:
		raise ValueError("signal_events cannot be empty")
	if not all(isinstance(event, dict) for event in signal_events):
		raise TypeError("signal_events must be a list (or sequence) of dictionaries")

	keys = list(signal_events[0].keys())
	expected = set(keys)
	for idx, event in enumerate(signal_events):
		current = set(event.keys())
		if current != expected:
			missing = sorted(expected - current)
			extra = sorted(current - expected)
			raise ValueError(
				f"All events must have the same keys. Event {idx} differs. "
				f"Missing={missing}, Extra={extra}"
			)
	return keys


def save_signal_events_parquet(
	signal_events: Sequence[Dict],
	output_path: str | Path,
	compression: str = "zstd",
) -> Path:
	"""Save signal events to a compressed columnar parquet file.

	The input is a list of dictionaries. Tensor/array fields are flattened into
	multiple parquet columns and reconstructed on load.
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	keys = _validate_signal_events(signal_events)
	n_events = len(signal_events)

	table_data = {}
	metadata = {
		"format": "nugget-signal-events-parquet",
		"version": 1,
		"num_events": n_events,
		"columns": {},
	}

	for key in keys:
		values = [event[key] for event in signal_events]
		first_value = values[0]

		if torch.is_tensor(first_value) or isinstance(first_value, np.ndarray):
			kind = "tensor" if torch.is_tensor(first_value) else "ndarray"
			arrays = []
			for value in values:
				array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
				arrays.append(array)

			reference_shape = arrays[0].shape
			if any(array.shape != reference_shape for array in arrays):
				raise ValueError(
					f"Field '{key}' has inconsistent shapes across events; "
					"cannot store in fixed columnar format"
				)

			stacked = np.stack(arrays, axis=0)
			flat = stacked.reshape(n_events, -1)

			for idx in range(flat.shape[1]):
				table_data[f"{key}__{idx}"] = flat[:, idx]

			metadata["columns"][key] = {
				"kind": kind,
				"shape": list(reference_shape),
				"dtype": str(stacked.dtype),
				"flat_size": int(flat.shape[1]),
			}
		else:
			scalar_values = [value.item() if isinstance(value, np.generic) else value for value in values]
			table_data[key] = scalar_values
			metadata["columns"][key] = {
				"kind": "scalar",
				"dtype": str(type(scalar_values[0]).__name__),
			}

	dataframe = pd.DataFrame(table_data)
	dataframe.to_parquet(output_path, index=False, compression=compression)

	with open(_metadata_path(output_path), "w", encoding="utf-8") as file_handle:
		json.dump(metadata, file_handle)

	return output_path


def _decode_events_from_dataframe(dataframe: pd.DataFrame, metadata: Dict) -> List[Dict]:
	n_events = len(dataframe)
	events: List[Dict] = [{} for _ in range(n_events)]

	for key, info in metadata["columns"].items():
		kind = info["kind"]

		if kind == "scalar":
			values = dataframe[key].tolist()
			for idx in range(n_events):
				events[idx][key] = values[idx]
			continue

		flat_size = int(info["flat_size"])
		shape = tuple(info["shape"])
		dtype = np.dtype(info["dtype"])
		column_names = [f"{key}__{i}" for i in range(flat_size)]

		matrix = dataframe[column_names].to_numpy(copy=False)
		matrix = matrix.astype(dtype, copy=False)
		values = matrix.reshape(n_events, *shape)

		for idx in range(n_events):
			value = np.array(values[idx], copy=True)
			if kind == "tensor":
				events[idx][key] = torch.from_numpy(value)
			else:
				events[idx][key] = value

	return events


def load_signal_events_parquet(input_path: str | Path) -> List[Dict]:
	"""Load signal events from parquet and return a list of dictionaries."""
	input_path = Path(input_path)
	meta_path = _metadata_path(input_path)

	with open(meta_path, "r", encoding="utf-8") as file_handle:
		metadata = json.load(file_handle)

	dataframe = pd.read_parquet(input_path)
	events = _decode_events_from_dataframe(dataframe, metadata)

	expected_count = int(metadata["num_events"])
	if len(events) != expected_count:
		raise ValueError(
			f"Loaded {len(events)} events but metadata expected {expected_count}"
		)
	return events


def iter_signal_events_parquet(input_path: str | Path, batch_size: int = 10000) -> Iterator[List[Dict]]:
	"""Iterate over signal events in batches to reduce peak memory usage.

	Requires `pyarrow` (used under pandas parquet support in this project).
	"""
	input_path = Path(input_path)
	meta_path = _metadata_path(input_path)

	with open(meta_path, "r", encoding="utf-8") as file_handle:
		metadata = json.load(file_handle)

	pq = importlib.import_module("pyarrow.parquet")

	parquet_file = pq.ParquetFile(input_path)
	for batch in parquet_file.iter_batches(batch_size=batch_size):
		dataframe = batch.to_pandas()
		yield _decode_events_from_dataframe(dataframe, metadata)

def _as_flat_tensor(value: Any) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return value.detach().reshape(-1)
	return torch.as_tensor(value).reshape(-1)


def _matches_zenith_limit(event_value: Any, limit_value: Any) -> bool | None:
	if not isinstance(limit_value, str):
		return None

	normalized_limit = limit_value.strip().lower()
	if "horizontal" in normalized_limit:
		threshold = torch.cos(torch.tensor(70*torch.pi/180.0))  # cos(70°) ≈ 0.342
		comparison = torch.lt
	elif "vertical" in normalized_limit:
		threshold = 0.8
		comparison = torch.gt
	else:
		return None

	zenith_tensor = _as_flat_tensor(event_value).to(dtype=torch.float32)
	abs_cos_zenith = torch.abs(torch.cos(zenith_tensor))
	return bool(torch.all(comparison(abs_cos_zenith, torch.as_tensor(threshold, dtype=abs_cos_zenith.dtype))))


def _matches_limit(event_key: str, event_value: Any, limit_value: Any) -> bool:
	if event_key == "zenith":
		zenith_match = _matches_zenith_limit(event_value, limit_value)
		if zenith_match is not None:
			return zenith_match

	event_tensor = _as_flat_tensor(event_value)

	if isinstance(limit_value, torch.Tensor):
		limit_value = limit_value.detach().cpu().tolist()

	if isinstance(limit_value, Sequence) and not isinstance(limit_value, (str, bytes)):
		if len(limit_value) == 2 and not any(
			isinstance(item, Sequence) and not isinstance(item, (str, bytes))
			for item in limit_value
		):
			lower = torch.as_tensor(limit_value[0], dtype=event_tensor.dtype)
			upper = torch.as_tensor(limit_value[1], dtype=event_tensor.dtype)
			return bool(torch.all(event_tensor >= lower) and torch.all(event_tensor <= upper))

		if len(limit_value) == event_tensor.numel():
			for event_item, item_limit in zip(event_tensor, limit_value):
				if isinstance(item_limit, Sequence) and not isinstance(item_limit, (str, bytes)):
					if len(item_limit) != 2:
						raise ValueError(
							"Per-component limits must be length-2 sequences of (min, max)."
						)
					lower = torch.as_tensor(item_limit[0], dtype=event_tensor.dtype)
					upper = torch.as_tensor(item_limit[1], dtype=event_tensor.dtype)
					if not bool((event_item >= lower) and (event_item <= upper)):
						return False
				else:
					if not bool(torch.isclose(event_item, torch.as_tensor(item_limit, dtype=event_tensor.dtype))):
						return False
			return True

	limit_tensor = _as_flat_tensor(limit_value).to(dtype=event_tensor.dtype)
	if limit_tensor.numel() == 1:
		return bool(torch.all(event_tensor == limit_tensor.item()))
	return bool(torch.equal(event_tensor, limit_tensor))


def select_event_indices(
	events: Sequence[Mapping[str, Any]],
	limits: Mapping[str, Any],
) -> list[int]:
	"""Return the indices of events that satisfy all provided limits.

	Parameters
	----------
	events:
		List of event dictionaries, such as the dictionaries returned by
		:class:`~nugget.samplers.cyl_sampler.CylinderSampler`.
	limits:
		Mapping from event keys to either exact values or inclusive bounds.
		Scalars may be given as ``[min, max]`` or ``(min, max)``. Vector values
		can use either a shared ``[min, max]`` bound for every component or a
		per-component sequence of ``[(min, max), ...]``.

	Returns
	-------
	list[int]
		Indices of the events that match the requested limits.
	"""
	selected_indices = []

	for event_index, event in enumerate(events):
		matches = True
		for key, limit_value in limits.items():
			if key not in event:
				matches = False
				break
			if not _matches_limit(key, event[key], limit_value):
				matches = False
				break

		if matches:
			selected_indices.append(event_index)

	return selected_indices


def select_events(
	events: Sequence[Mapping[str, Any]],
	limits: Mapping[str, Any],
) -> list[dict[str, Any]]:
	"""Return the filtered events in the same order they appear in `events`.

	This uses the same limit semantics as :func:`select_event_indices`.
	"""
	selected_events = []

	for event in events:
		matches = True
		for key, limit_value in limits.items():
			if key not in event:
				matches = False
				break
			if not _matches_limit(key, event[key], limit_value):
				matches = False
				break

		if matches:
			selected_events.append(dict(event))

	return selected_events

