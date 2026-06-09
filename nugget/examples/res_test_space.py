import argparse
from pathlib import Path

import nugget  # Main NUGGET package for neutrino detector optimization
import torch
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fisher information on a shard of signal events.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-events", type=int, default=50000)
    parser.add_argument("--version", default="r600_50_u_1")
    parser.add_argument("--signal-events-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("res_test/space_test"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--spacing-min", type=float, default=25.0)
    parser.add_argument("--spacing-max", type=float, default=200.0)
    parser.add_argument("--spacing-count", type=int, default=20)
    parser.add_argument("--use_llrnet", type=str, default="true")
    return parser.parse_args()


def move_events_to_device(signal_events, device):
    moved_events = []
    for signal_event in signal_events:
        moved_event = {}
        for key, value in signal_event.items():
            moved_event[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        moved_events.append(moved_event)
    return moved_events


def split_events(signal_events, num_shards, shard_index):
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    event_indices = np.arange(len(signal_events))
    shard_event_indices = np.array_split(event_indices, num_shards)[shard_index]
    shard_events = [signal_events[idx] for idx in shard_event_indices.tolist()]
    return shard_event_indices, shard_events


args = parse_args()

use_llrnet = args.use_llrnet == "true"
device = args.device
num_events = args.num_events
version = args.version
output_dir = args.output_dir

signal_events_path = args.signal_events_path or Path(f"res_test/signal_events_{num_events}_{version}.pt")

print(f"Using device: {device}")
print(f"Using signal_version: {version}")
print(f"Using signal events path: {signal_events_path}")
print(f"Using LLRnet surrogate: {use_llrnet}")

if use_llrnet:
    extra = ""
else:
    extra = "_lambda"
version += extra
output_dir = output_dir / f"{version}"
output_dir.mkdir(parents=True, exist_ok=True)
center = [0, 0, 0]
radius = 600
height = 1000
lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabre(device=device, use_poisson=use_llrnet, domain_size=1600, particle_mode='track')
light_yield_surrogate = lightsabre_surrogate.light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
                                                    device=None,
                                                    event_type='signal',
                                                    domain_size=1600,
                                                    E_min=1e2,
                                                    E_max=1e8,
                                                    find_exact_intersection=False,
                                                    random_position_along_ray=True,
                                                    energy_dist='log_uniform',
                                                    cylinder_center=center,
                                                    cylinder_radius=radius,
                                                    cylinder_height=height,
                                                    # point_towards_center=True,
                                                    # cos_range=torch.tensor((np.cos(np.radians(155)),np.cos(np.radians(180))))
                                                    )
# signal_events = signal_sampler.sample_events(num_events)
signal_events = nugget.utils.data_tools.load_signal_events_parquet(signal_events_path)

shard_event_indices, signal_events = split_events(signal_events, args.num_shards, args.shard_index)
signal_events = move_events_to_device(signal_events, device)

if len(shard_event_indices) > 0:
    shard_start = int(shard_event_indices[0])
    shard_stop = int(shard_event_indices[-1]) + 1
else:
    shard_start = 0
    shard_stop = 0

print(
    f"Using shard {args.shard_index + 1}/{args.num_shards}: "
    f"events {shard_start}:{shard_stop} ({len(signal_events)} events)"
)

for string_spacing in np.linspace(args.spacing_min, args.spacing_max, args.spacing_count):
    print(f"Running fisher info calculations for string spacing: {string_spacing}m")
    geometry = nugget.geometries.SpaceString.SpaceString(
            device=device,
            hex_type='hexagonal',
            domain_size=2500,
            dim=3,
            n_strings=61,
            points_per_string=20,
            starting_spacing=string_spacing,
            starting_z_spacing=50
        )
    geom_dict = geometry.initialize_points()

    angular_resolution_loss = nugget.losses.fisher_info.WeightedResolutionLoss(
        device=device,
        resolution_type='angular',
        fisher_info_params=['position', 'energy', 'direction']
    )

    signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(
        device=device,
    )
    if use_llrnet:
        llr_net = nugget.surrogates.LLRnet.LLRnet(
        device=device,
        domain_size=2500,
        dim=3,
        hidden_dims=[64, 64, 64, 64, 64, 64],
        use_fourier_features=False,
        num_parallel_branches=1,
        learnable_frequencies=False,
        dropout_rate=0.05,
        learning_rate=3e-4,
        shared_mlp=False,
        use_residual_connections=True,
        signal_noise_scale=0,
        background_noise_scale=0,
        add_relative_pos=False,
        log_scale_ly=True,
        norm_pos=True,
        log_scale_energy=True,
        reduce_lr_on_plateau=True,
        lr_scheduler_patience=30,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
        min_photons=1,
        num_photons_per_sample=None,
        input_charge=False,
        rel_time=False,
        input_delta_time=False,
        use_rich_features=True,
        add_distance_from_beam=False,
        use_patd=False,
        )

        llr_net.load_model('best_charge_llr_model_v5.pt')

    fisher_info_per_string_per_event = angular_resolution_loss.compute_fisher_info_per_string_per_event(
                string_xy=geom_dict['string_xy'],
                points_3d=geom_dict['points_3d'],
                signal_event_params=signal_events,
                signal_surrogate_func=light_yield_surrogate,
                llr_net=llr_net if use_llrnet else None,
                llr_iterations=100,
                skip_zero_response=True,
                verbose=True,
                jacrev_chunk_size=50000,
                point_chunk_size=11000,
                grad_chunk_size=7,
                llr_autodiff_mode='jvp'
                )

    output_path = output_dir / (
        f"fisher_info_per_string_per_event_{num_events}_{int(string_spacing)}_{version}"
        f"_shard{args.shard_index:03d}of{args.num_shards:03d}"
        f"_events{shard_start:06d}-{shard_stop:06d}.pt"
    )
    torch.save(fisher_info_per_string_per_event.cpu(), output_path)


# loss_params = {
# # 'cylinder_kwargs': {'find_exact_intersection': True},
# # 'num_events_per_bin': 10,
# 'signal_surrogate_func': light_yield_surrogate,
# 'bounding_cylinder_temperature': 10,
# # 'perfect_efficiency': False,
# # 'zenith_range': (-1, 0),
# # 'skip_zero_response': True,
# 'fisher_info_llr_net': llr_net,
# 'use_relative_energy': True,
# 'signal_event_params': signal_events,
# 'fisher_info_llr_iterations': 100,
# 'llr_event_labels': ['position','energy', 'direction'],
# }
    