import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define operators (same as script 1) ---

def op_at_site(op, site, N):
    ops = [qt.qeye(2) for _ in range(N)]
    ops[site] = op
    return qt.tensor(ops)

def n_op(site, N):
    return op_at_site(qt.num(2), site, N)

def sigmax_op(site, N):
    return op_at_site(qt.sigmax(), site, N)

N = 7
Omega = 1.0  
tilde_Omega = (np.sqrt(2) * Omega) / 2 
Vr1_O = 20.0
Vr2_O = 10.0
dD1_O = -0.5
dD2_O = 0.5
T_pulse = np.pi / tilde_Omega 
Vr1 = Vr1_O * Omega
Vr2 = Vr2_O * Omega
Delta1 = Vr1
Delta2 = Vr2 

gauss_sigma = T_pulse 

# --- 3. Define time-dependent Hamiltonian ---

# Static part H_int
H_int = 0
for j in range(0, N - 1, 2):
    H_int += Vr1 * n_op(j, N) * n_op(j + 1, N)
for j in range(1, N - 1, 2):
    H_int += Vr2 * n_op(j, N) * n_op(j + 1, N)

# Detuning term
H_det_op = sum(n_op(j, N) for j in range(N))

# Driving term (bare)
H_drive_bare = sum(sigmax_op(j, N) for j in range(N)) * (1.0 / 2.0)

# Define Gaussian pulse function (coefficient)
# Paper suggests using ramped or shaped pulses [cite: 291]
def gaussian_pulse_shape(t, args):
    """
    Define a Gaussian pulse centered at T_pulse/2.
    Peak intensity is Omega_peak.
    """
    T = args['T']
    Omega_peak = args['Omega_peak']
    sigma = args['sigma']
    return Omega_peak * np.exp(-(t - T/2)**2 / (2 * sigma**2))

# --- 4. Set initial state and evolution ---

psi0_list = [qt.basis(2, 0) for _ in range(N)]
psi0_list[0] = qt.basis(2, 1) # |1000000>
psi0 = qt.tensor(psi0_list)
n_ops_list = [n_op(j, N) for j in range(N)]
num_steps = 10
t_list_step = np.linspace(0, T_pulse, 21) 

gauss_args = {'T': T_pulse, 'sigma': gauss_sigma, 'Omega_peak': Omega}

# --- Print Gaussian pulse shape (sampling points) ---
t_samples = np.linspace(0.0, T_pulse, 21)
gauss_vals = [gaussian_pulse_shape(t, gauss_args) for t in t_samples]

# Store results
current_psi = psi0
all_populations_g = []
all_times_g = []

pop0 = qt.expect(n_ops_list, current_psi)
all_populations_g.append(pop0)
all_times_g.append(0.0)

opts = qt.Options(store_states=True)

for i in range(num_steps):
    # Select detuning
    Delta_i = Delta1 if i % 2 == 0 else Delta2
    
    # 1. Static part (H_int - Delta * H_det)
    H_static = H_int - Delta_i * H_det_op
    
    # 2. Time-dependent part ([H_drive_bare, coeff_func])
    H_dynamic = [H_drive_bare, gaussian_pulse_shape]
    
    # 3. Combine into QuTiP list
    H_t = [H_static, H_dynamic]
    
    # 4. Solve (note: pass args)
    result = qt.sesolve(H_t, current_psi, t_list_step, 
                        e_ops=n_ops_list, args=gauss_args, options=opts)
    
    # (Store results, same as above)
    current_psi = result.states[-1]
    current_time_offset = all_times_g[-1]
    step_pops = [res[1:] for res in result.expect]
    step_times = t_list_step[1:] + current_time_offset
    
    if i == 0:
        all_populations_g = np.array(step_pops)
        all_times_g = step_times
    else:
        all_populations_g = np.hstack((all_populations_g, step_pops))
        all_times_g = np.hstack((all_times_g, step_times))
        
# Add initial time point
all_times_g = np.insert(all_times_g, 0, 0.0)
all_populations_g = np.hstack((np.array(pop0).reshape(-1, 1), all_populations_g))

# --- 5. Visualize results (Gaussian version) ---

time_axis_plot_g = all_times_g * tilde_Omega / (2 * np.pi)

# Construct Gaussian pulse waveform sequence corresponding to the heatmap time axis
pulse_vals = []
# The first point corresponds to the initial state, pulse value is 0
pulse_vals.append(0.0)
# Then add the corresponding Gaussian pulse value step by step
for i in range(num_steps):
    # Each step has t_list_step[1:] time points
    for t_rel in t_list_step[1:]:
        pulse_val = gaussian_pulse_shape(t_rel, gauss_args)
        pulse_vals.append(pulse_val)
pulse_vals = np.array(pulse_vals)

fig, ax1 = plt.subplots(figsize=(7, 4))

ax1.pcolormesh(time_axis_plot_g, np.arange(N), all_populations_g,
                cmap='viridis', vmin=0, vmax=1)
ax1.set_xlabel(r"Time $\tilde{\Omega}t / (2\pi)$", fontsize=20)
ax1.set_ylabel("Atom Site", fontsize=20)
ax1.set_yticks(np.arange(N))
ax1.set_yticklabels([f"{j+1}" for j in range(N)], fontsize=20)
ax1.set_ylim(-0.5, N - 0.5)

# Calculate and label the maximum fidelity of propagation to site 7
target_site = N - 1  # The index of site 7 is 6
fids_site7 = all_populations_g[target_site, :]
idx_max = int(np.argmax(fids_site7))
fid_max = float(fids_site7[idx_max])
t_max = float(all_times_g[idx_max])
t_max_plot = float(time_axis_plot_g[idx_max])

ax1.axvline(t_max_plot, color='w', linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig('pulse_optimization_1d.svg')