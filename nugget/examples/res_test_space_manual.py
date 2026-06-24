from pathlib import Path

import nugget  # Main NUGGET package for neutrino detector optimization
import torch
import numpy as np



def move_events_to_device(signal_events, device):
    moved_events = []
    for signal_event in signal_events:
        moved_event = {}
        for key, value in signal_event.items():
            moved_event[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        moved_events.append(moved_event)
    return moved_events

min_energy=6
max_energy=8
spacing_min=25.0
spacing_max=300.0
spacing_count=25
use_llrnet = True
use_patd = False
device = "cuda:2"
num_events = 1000
zenith_range = 'horizontal'
version = f"r600_50_u_sp_1"
output_dir = Path("res_test/space_test")

signal_events_path = Path(f"res_test/space_test/{version}/signal_events_{num_events}_e{min_energy}-{max_energy}_{zenith_range}.pt")

print(f"Using device: {device}")
print(f"Using signal_version: {version}")
print(f"Using signal events path: {signal_events_path}")
print(f"Using LLRnet surrogate: {use_llrnet}")
print(f"Using PATD surrogate: {use_patd}")
if use_llrnet:
    extra = ""
else:
    extra = "_lambda"
version += extra
if use_patd:
    version += "_patd"
output_dir = output_dir / f"{version}"
output_dir.mkdir(parents=True, exist_ok=True)
center = [0, 0, 0]
radius = 600
height = 1000

if not use_patd:
    lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabre(device=device, use_poisson=False, domain_size=2500, particle_mode = 'track')
else:
    lightsabre_surrogate = nugget.surrogates.LightSabre.LightSabrePATD(
        device=device,
        use_poisson=False,
        num_track_points=1000,
        domain_size=2500,
        use_max_energy_dist=True,
        use_perpendicular_distance_only=True,
        particle_mode='track',
        )
light_yield_surrogate = lightsabre_surrogate.light_yield_surrogate
signal_sampler = nugget.samplers.cyl_sampler.CylinderSampler(
                                                    device=None,
                                                    event_type='signal',
                                                    domain_size=2500,
                                                    E_min=10**min_energy,
                                                    E_max=10**max_energy,
                                                    find_exact_intersection=True,
                                                    random_position_along_ray=False,
                                                    energy_dist='log_uniform',
                                                    uniform_zenith_sampling=True,
                                                    cylinder_center=center,
                                                    cylinder_radius=radius,
                                                    cylinder_height=height,
                                                    # point_towards_center=True,
                                                    cos_range=zenith_range if zenith_range in ['horizontal', 'vertical'] else torch.tensor([-1.0, 1.0]),
                                                    )
signal_events = signal_sampler.sample_events(num_events)
nugget.utils.data_tools.save_signal_events_parquet(signal_events, signal_events_path)
signal_events = move_events_to_device(signal_events, device)
for string_spacing in np.linspace(spacing_min, spacing_max, spacing_count):
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
        fisher_info_params=['energy', 'direction']
    )

    signal_yield_loss_func = nugget.losses.light_yield.WeightedLightYieldLoss(
        device=device,
    )
    if use_llrnet:
        llr_net = nugget.surrogates.LLRnet.LLRnet(
        device=device,
        domain_size=2000,
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
        add_distance_from_beam=use_patd,
        use_patd=use_patd,
        )
        if not use_patd:
            llr_net.load_model('best_charge_llr_model_v5.pt')
        else:
            llr_net.load_model('best_hit_llr_model_v7.pt')

    fisher_info_per_string_per_event = angular_resolution_loss.compute_fisher_info_per_string_per_event(
                string_xy=geom_dict['string_xy'],
                points_3d=geom_dict['points_3d'],
                signal_event_params=signal_events,
                signal_surrogate_func=light_yield_surrogate,
                llr_net=llr_net if use_llrnet else None,
                llr_iterations=200,
                skip_zero_response=True,
                verbose=True,
                jacrev_chunk_size=50000,
                point_chunk_size=11000,
                grad_chunk_size=7,
                llr_autodiff_mode='jvp',
                use_patd=use_patd,
                use_rich_features=True,
                use_patd_quadrature=use_patd,
                t_offset_ns=0.0,
                t_max_ns=5000.0,
                zero_response_threshold=0.01,
                use_charge_quadrature=not use_patd,  # for patd, charge quadrature is handled inside the surrogate
                charge_center_on_llr_peak=not use_patd,  # for patd, the surrogate is already centered on the llr peak
                charge_peak_scan_points=64,
                adaptive_grid_retry=True,
                adaptive_t_max_floor_ns=10,
                uninformative_fisher_value=1e-6,
                )

    output_path = output_dir / (
        f"fisher_info_per_string_per_event_{num_events}_{int(string_spacing)}_e{min_energy}-{max_energy}_{zenith_range}.pt"
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
    