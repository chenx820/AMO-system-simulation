#!/usr/bin/env python3
"""
Detuning Optimization Script
optimize the detuning parameters of the AMO system

Author: Chen Huang
Date: 2025-08-22
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

from src.transport_2d import run_simulation
from src.system_params import create_custom_system


def build_base_params(Nx, Ny, Omega):
    """
    build the base parameters using the system parameters class
    return params, pi_pulse_duration
    """
    # create the system parameters object
    system_params = create_custom_system(Nx, Ny, Omega_MHz=Omega/(2*np.pi*1e6))
    
    # reset the detuning to 0 (for optimization)
    system_params.update_detuning(0, 0, 0, 0)
    
    return system_params.get_params_dict(), system_params.pi_pulse_duration


def make_pulse_sequence(periods, pi_pulse_duration, steps_per_pulse):
    """
    generate the pulse sequence consistent with transport_2d.py: [('h1', T_pi, steps), ('h2', T_pi, steps)] * periods
    """
    pulse_sequence = []
    for _ in range(periods):
        pulse_sequence.extend([
            ('h1', pi_pulse_duration, steps_per_pulse),
            ('h2', pi_pulse_duration, steps_per_pulse),
        ])
    return pulse_sequence


def evaluate_fidelity(Nx, Ny, base_params, pulse_sequence, initial_site, target_site,
                      delta_h1, delta_h2):
    """
    Legacy: evaluate for (h1,h2) only.
    """
    params = deepcopy(base_params)
    params['delta_detuning_map']['h1'] = delta_h1
    params['delta_detuning_map']['h2'] = delta_h2

    times, history = run_simulation(Nx, Ny, params, pulse_sequence, [initial_site])
    if len(history) == 0:
        return -1.0
    final_pop = history[-1]
    return float(final_pop[target_site])


def evaluate_fidelity_pair(Nx, Ny, base_params, pulse_sequence, initial_site, target_site,
                           key1, key2, val1, val2):
    """
    Generalized two-parameter evaluation for keys in {'h1','h2','v1','v2'}.
    """
    params = deepcopy(base_params)
    params['delta_detuning_map'][key1] = val1
    params['delta_detuning_map'][key2] = val2
    times, history = run_simulation(Nx, Ny, params, pulse_sequence, [initial_site])
    if len(history) == 0:
        return -1.0
    final_pop = history[-1]
    return float(final_pop[target_site])


def grid_search(Nx, Ny, Omega, initial_site, target_site,
                periods=3, steps_per_pulse=40,
                delta_range=0.3, coarse_steps=25, fine_window=0.06, fine_steps=21,
                heatmap_path='./outputs/plots/delta_grid_heatmap.png'):
    """
    two-stage grid search:
      1) coarse scan delta_h1, delta_h2 ∈ [-delta_range, +delta_range] * Omega
      2) fine scan around the best point ±fine_window * Omega
    output: the best (delta_h1, delta_h2) and save the coarse scan heatmap
    """
    params, pi_pulse_duration = build_base_params(Nx, Ny, Omega)
    pulse_sequence = make_pulse_sequence(periods, pi_pulse_duration, steps_per_pulse)

    # coarse scan
    rel_vals = np.linspace(-delta_range, delta_range, coarse_steps)
    d1_list = rel_vals * Omega
    d2_list = rel_vals * Omega

    heatmap = np.zeros((len(d1_list), len(d2_list)))
    best_fid = -1.0
    best_d1 = None
    best_d2 = None

    for i, d1 in enumerate(d1_list):
        for j, d2 in enumerate(d2_list):
            fid = evaluate_fidelity(Nx, Ny, params, pulse_sequence, initial_site, target_site, d1, d2)
            heatmap[i, j] = fid
            if fid > best_fid:
                best_fid, best_d1, best_d2 = fid, d1, d2

    # save the coarse scan heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap.T, origin='lower', cmap='viridis',
                    extent=[d1_list[0]/Omega, d1_list[-1]/Omega, d2_list[0]/Omega, d2_list[-1]/Omega],
                    aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='final population at target')
    plt.xlabel(r'$\delta_{h1} / \Omega$')
    plt.ylabel(r'$\delta_{h2} / \Omega$')
    plt.title('Coarse grid search')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=800)
    plt.close()

    # fine scan
    fine_vals = np.linspace(-fine_window, fine_window, fine_steps) * Omega
    fine_d1_list = best_d1 + fine_vals
    fine_d2_list = best_d2 + fine_vals

    fine_best_fid = best_fid
    fine_best_d1 = best_d1
    fine_best_d2 = best_d2

    for d1 in fine_d1_list:
        for d2 in fine_d2_list:
            fid = evaluate_fidelity(Nx, Ny, params, pulse_sequence, initial_site, target_site, d1, d2)
            if fid > fine_best_fid:
                fine_best_fid, fine_best_d1, fine_best_d2 = fid, d1, d2

    return {
        'coarse_best': (best_d1, best_d2, best_fid),
        'fine_best': (fine_best_d1, fine_best_d2, fine_best_fid),
        'pi_pulse_duration': pi_pulse_duration,
    }


def grid_search_pair(Nx, Ny, Omega, initial_site, target_site,
                     pair=('h1','h2'),
                     periods=1, steps_per_pulse=15,
                     delta_range=0.3, coarse_steps=21,
                     fine_window=0.05, fine_steps=15):
    """
    Two-stage grid search for an arbitrary pair among (h1,h2,v1,v2).
    Returns (best_key1_val, best_key2_val, best_fidelity).
    """
    key1, key2 = pair
    base_params, pi_pulse_duration = build_base_params(Nx, Ny, Omega)
    pulse_sequence = make_pulse_sequence(periods, pi_pulse_duration, steps_per_pulse)

    rel_vals = np.linspace(-delta_range, delta_range, coarse_steps)
    list1 = rel_vals * Omega
    list2 = rel_vals * Omega

    best_fid = -1.0
    best_v1 = 0.0
    best_v2 = 0.0
    for v1 in list1:
        for v2 in list2:
            fid = evaluate_fidelity_pair(Nx, Ny, base_params, pulse_sequence, initial_site, target_site,
                                         key1, key2, v1, v2)
            if fid > best_fid:
                best_fid, best_v1, best_v2 = fid, v1, v2

    fine_vals = np.linspace(-fine_window, fine_window, fine_steps) * Omega
    fine_list1 = best_v1 + fine_vals
    fine_list2 = best_v2 + fine_vals

    fine_best_fid = best_fid
    fine_best_v1 = best_v1
    fine_best_v2 = best_v2
    for v1 in fine_list1:
        for v2 in fine_list2:
            fid = evaluate_fidelity_pair(Nx, Ny, base_params, pulse_sequence, initial_site, target_site,
                                         key1, key2, v1, v2)
            if fid > fine_best_fid:
                fine_best_fid, fine_best_v1, fine_best_v2 = fid, v1, v2

    return (fine_best_v1, fine_best_v2, fine_best_fid)


def optimize_all_deltas_alternating(Nx, Ny, Omega, initial_site, target_site, *,
                                    delta_init: dict | None = None,
                                    alternating_max_rounds: int = 3,
                                    convergence_eps_rel: float = 0.001,
                                    low_fidelity: dict = None,
                                    mid_fidelity: dict = None,
                                    high_fidelity: dict = None):
    """
    Alternating 2D optimization over (h1,h2) and (v1,v2) with multi-fidelity.
    Returns dict: { 'h1':..., 'h2':..., 'v1':..., 'v2':..., 'fidelity': best_seen }
    """
    # initialize
    if delta_init is None:
        delta_init = {'h1': 0.0, 'h2': 0.0, 'v1': 0.0, 'v2': 0.0}
    deltas = dict(delta_init)

    def norm_rel_change(old, new):
        denom = max(1e-12, max(abs(val) for val in old.values()))
        num = sum(abs(new[k]-old[k]) for k in new.keys())
        return num / denom

    best_fid_seen = -1.0
    # low → mid rounds; final high validation per pair
    fidelity_schedules = [low_fidelity or {'periods':1,'steps_per_pulse':12},
                          mid_fidelity or {'periods':1,'steps_per_pulse':18}]

    for round_idx in range(alternating_max_rounds):
        prev = deltas.copy()
        schedule = fidelity_schedules[min(round_idx, len(fidelity_schedules)-1)]

        # optimize (h1,h2)
        h1_val, h2_val, fid_h = grid_search_pair(
            Nx, Ny, Omega, initial_site, target_site, pair=('h1','h2'),
            periods=schedule['periods'], steps_per_pulse=schedule['steps_per_pulse'])
        deltas['h1'], deltas['h2'] = h1_val, h2_val
        best_fid_seen = max(best_fid_seen, fid_h)

        # optimize (v1,v2)
        v1_val, v2_val, fid_v = grid_search_pair(
            Nx, Ny, Omega, initial_site, target_site, pair=('v1','v2'),
            periods=schedule['periods'], steps_per_pulse=schedule['steps_per_pulse'])
        deltas['v1'], deltas['v2'] = v1_val, v2_val
        best_fid_seen = max(best_fid_seen, fid_v)

        change = norm_rel_change(prev, deltas)
        if change < convergence_eps_rel:
            break

    # final high-fidelity polish per pair (optional)
    if high_fidelity:
        h1_val, h2_val, fid_h = grid_search_pair(
            Nx, Ny, Omega, initial_site, target_site, pair=('h1','h2'),
            periods=high_fidelity['periods'], steps_per_pulse=high_fidelity['steps_per_pulse'])
        deltas['h1'], deltas['h2'] = h1_val, h2_val
        best_fid_seen = max(best_fid_seen, fid_h)

        v1_val, v2_val, fid_v = grid_search_pair(
            Nx, Ny, Omega, initial_site, target_site, pair=('v1','v2'),
            periods=high_fidelity['periods'], steps_per_pulse=high_fidelity['steps_per_pulse'])
        deltas['v1'], deltas['v2'] = v1_val, v2_val
        best_fid_seen = max(best_fid_seen, fid_v)

    return {
        'deltas': deltas,
        'fidelity': best_fid_seen,
    }


def optimize_delta_1d(Nx, Ny, Omega, direction, initial_site, target_site, *,
                      periods=1, steps_per_pulse=15,
                      delta_range=0.3, coarse_steps=21,
                      fine_window=0.05, fine_steps=15):
    """
    1D coarse+fine scan for a single direction's detuning.
    Returns best_delta (rad/s) and best_fidelity.
    """
    base_params, pi_pulse_duration = build_base_params(Nx, Ny, Omega)
    pulse_sequence = [(direction, pi_pulse_duration, steps_per_pulse)] * periods

    rel_vals = np.linspace(-delta_range, delta_range, coarse_steps)
    delta_list = rel_vals * Omega

    best_delta = 0.0
    best_fid = -1.0
    for d in delta_list:
        params = deepcopy(base_params)
        params['delta_detuning_map'][direction] = d
        times, history = run_simulation(Nx, Ny, params, pulse_sequence, [initial_site])
        fid = float(history[-1][target_site]) if len(history) else -1.0
        if fid > best_fid:
            best_fid, best_delta = fid, d

    fine_vals = np.linspace(-fine_window, fine_window, fine_steps) * Omega
    for off in fine_vals:
        d = best_delta + off
        params = deepcopy(base_params)
        params['delta_detuning_map'][direction] = d
        times, history = run_simulation(Nx, Ny, params, pulse_sequence, [initial_site])
        fid = float(history[-1][target_site]) if len(history) else -1.0
        if fid > best_fid:
            best_fid, best_delta = fid, d

    return best_delta, best_fid, pi_pulse_duration


def optimize_path_stepwise(Nx, Ny, Omega, *,
                           steps: list,
                           start_site: int,
                           periods: int = 1,
                           steps_per_pulse: int = 15,
                           delta_range: float = 0.3,
                           coarse_steps: int = 21,
                           fine_window: float = 0.05,
                           fine_steps: int = 15):
    """
    steps: list of tuples (direction, target_site)
    For each step, optimize only that direction's detuning to maximize population at the step target.
    Returns: list of pulse tuples (direction, duration, steps_per_pulse, delta_override)
    """
    pulse_list = []
    current_site = start_site
    # duration will be T_pi from the first call; assume same Omega
    for (direction, target_site) in steps:
        best_delta, best_fid, T_pi = optimize_delta_1d(
            Nx, Ny, Omega, direction, current_site, target_site,
            periods=periods, steps_per_pulse=steps_per_pulse,
            delta_range=delta_range, coarse_steps=coarse_steps,
            fine_window=fine_window, fine_steps=fine_steps
        )
        pulse_list.append((direction, T_pi, steps_per_pulse, best_delta))
        current_site = target_site
    return pulse_list


def main():
    parser = argparse.ArgumentParser(description='Grid search for detuning deltas (h1, h2).')
    parser.add_argument('--Nx', type=int, default=4)
    parser.add_argument('--Ny', type=int, default=4)
    parser.add_argument('--Omega_MHz', type=float, default=3.0, help='Rabi frequency in MHz (linear, not angular).')
    parser.add_argument('--periods', type=int, default=3)
    parser.add_argument('--steps_per_pulse', type=int, default=40)
    parser.add_argument('--delta_range', type=float, default=0.3, help='coarse range in units of Ω')
    parser.add_argument('--coarse_steps', type=int, default=25)
    parser.add_argument('--fine_window', type=float, default=0.06, help='fine window half-width in units of Ω')
    parser.add_argument('--fine_steps', type=int, default=21)
    parser.add_argument('--initial_site', type=int, default=0)
    parser.add_argument('--target_site', type=int, default=5)
    parser.add_argument('--heatmap', type=str, default='delta_grid_heatmap.png')
    args = parser.parse_args()

    Nx = args.Nx
    Ny = args.Ny
    Omega = 2 * np.pi * args.Omega_MHz * 1e6

    res = grid_search(
        Nx=Nx,
        Ny=Ny,
        Omega=Omega,
        initial_site=args.initial_site,
        target_site=args.target_site,
        periods=args.periods,
        steps_per_pulse=args.steps_per_pulse,
        delta_range=args.delta_range,
        coarse_steps=args.coarse_steps,
        fine_window=args.fine_window,
        fine_steps=args.fine_steps,
        heatmap_path=args.heatmap,
    )

    (c_d1, c_d2, c_fid) = res['coarse_best']
    (f_d1, f_d2, f_fid) = res['fine_best']

    print('--- Grid search results ---')
    print(f'Coarse best:  delta_h1={c_d1/Omega:+.3f}Omega, delta_h2={c_d2/Omega:+.3f}Omega, fidelity={c_fid:.4f}')
    print(f'Fine   best:  delta_h1={f_d1/Omega:+.3f}Omega, delta_h2={f_d2/Omega:+.3f}Omega, fidelity={f_fid:.4f}')
    print(f"Saved coarse heatmap to '{args.heatmap}'")


if __name__ == '__main__':
    main()


