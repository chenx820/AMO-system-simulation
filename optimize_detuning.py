import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

# 依赖本仓库中的动力学例程
from transport_2d import run_simulation


def build_base_params(Nx, Ny, Omega):
    """
    与 transport_2d.py 保持一致的物理与派生参数（除 detuning_map 以外）。
    返回 params, pi_pulse_duration。
    """
    Omega_eff = Omega / np.sqrt(2)

    # 与 transport_2d.py 相同的间距（可根据需要外部化）
    r_v1, r_v2 = 11.4e-6, 12.8e-6
    r_h1, r_h2 = 10.0e-6, 13.8e-6

    # C6 与相互作用
    C6 = 3e7 * Omega * (1e-6) ** 6
    V_map = {
        'h1': Omega * 20,
        'h2': Omega * 10,
        'v1': C6 / r_v1 ** 6,
        'v2': C6 / r_v2 ** 6,
    }

    # 初始 detuning 小修正（随后会被搜索覆盖）
    delta_detuning_map = {
        'h1': -0.133 * Omega,
        'h2': -0.033 * Omega,
        'v1': 0.0 * Omega,
        'v2': -0.2 * Omega,
    }

    params = {
        'mode': 'Schrodinger',
        'Omega': Omega,
        'V_map': V_map,
        'Gamma_decay': 0.0,
        'Gamma_dephasing': 0.0,
        'delta_detuning_map': delta_detuning_map,
    }

    pi_pulse_duration = np.pi / Omega_eff
    return params, pi_pulse_duration


def make_pulse_sequence(periods, pi_pulse_duration, steps_per_pulse):
    """
    生成与 transport_2d.py 一致的两段式脉冲序列：[('h1', Tπ, steps), ('h2', Tπ, steps)] × periods
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
    替换 detuning_map 中的 h1/h2 偏置，运行短时模拟并返回目标站点末态人口。
    """
    params = deepcopy(base_params)
    params['delta_detuning_map']['h1'] = delta_h1
    params['delta_detuning_map']['h2'] = delta_h2

    times, history = run_simulation(Nx, Ny, params, pulse_sequence, initial_site)
    if len(history) == 0:
        return -1.0
    final_pop = history[-1]
    return float(final_pop[target_site])


def grid_search(Nx, Ny, Omega, initial_site, target_site,
                periods=3, steps_per_pulse=40,
                delta_range=0.3, coarse_steps=25, fine_window=0.06, fine_steps=21,
                heatmap_path='delta_grid_heatmap.png'):
    """
    两阶段网格搜索：
      1) 粗扫 δ_h1, δ_h2 ∈ [-delta_range, +delta_range]·Ω
      2) 以最优点为中心做细扫窗口 ±fine_window·Ω
    输出：最优 (δ_h1, δ_h2) 以及保存粗扫热力图。
    """
    params, pi_pulse_duration = build_base_params(Nx, Ny, Omega)
    pulse_sequence = make_pulse_sequence(periods, pi_pulse_duration, steps_per_pulse)

    # 粗扫
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

    # 保存粗扫热力图
    plt.figure(figsize=(6, 5))
    im = plt.imshow(heatmap.T, origin='lower', cmap='viridis',
                    extent=[d1_list[0]/Omega, d1_list[-1]/Omega, d2_list[0]/Omega, d2_list[-1]/Omega],
                    aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='final population at target')
    plt.xlabel('δ_h1 / Ω')
    plt.ylabel('δ_h2 / Ω')
    plt.title('Coarse grid search')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=800)
    plt.close()

    # 细扫
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
    print(f'Coarse best:  δ_h1={c_d1/Omega:+.3f}Ω, δ_h2={c_d2/Omega:+.3f}Ω, fidelity={c_fid:.4f}')
    print(f'Fine   best:  δ_h1={f_d1/Omega:+.3f}Ω, δ_h2={f_d2/Omega:+.3f}Ω, fidelity={f_fid:.4f}')
    print(f"Saved coarse heatmap to '{args.heatmap}'")


if __name__ == '__main__':
    main()


