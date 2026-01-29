#!/usr/bin/env python3
import os
import yaml
import time
import numpy as np
import pybullet as p
import pybullet_data
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except Exception:
    PINOCCHIO_AVAILABLE = False
print(f"Pinocchio available: {PINOCCHIO_AVAILABLE}")

# Simple PyBullet simulator that reproduces dynamics and writes CSVs

def expand_user(path):
    return os.path.expanduser(path)


def main():
    cfg_path = expand_user('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # params under /**/ros__parameters
    params = cfg.get('/**', {}).get('ros__parameters', {})
    # fallback to top-level keys if not found
    q_init = params.get('q_init', 0.2)
    # allow q_init to be scalar or per-joint list; coerce to floats
    q_init_list = None
    try:
        if isinstance(q_init, (list, tuple)):
            q_init_list = [float(x) for x in q_init]
        else:
            q_init = float(q_init)
    except Exception:
        # leave as-is; downstream code will handle
        try:
            q_init = float(q_init)
        except Exception:
            pass
    timestep = params.get('timestep', 0.01)
    duration = params.get('duration', 10.0)
    urdf_path = expand_user(params.get('urdf_path', '~/ros2_ws/dynamic_ws/src/vi_2p/urdf/7_pendulum.urdf'))

    n_steps = int(duration / timestep)

    # Start PyBullet in DIRECT
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(timestep, physicsClientId=client)

    flags = p.URDF_USE_INERTIA_FROM_FILE
    body = p.loadURDF(urdf_path, useFixedBase=True, flags=flags, physicsClientId=client)

    num_joints = p.getNumJoints(body, physicsClientId=client)

    # Build Pinocchio model if available (used to include Pinocchio cost in timing
    # and to compute inertia matrix consistently)
    pin_model = None
    pin_data = None
    pin_error_count = 0
    pin_runtimes = []
    if PINOCCHIO_AVAILABLE:
        try:
            pin_model = pin.buildModelFromUrdf(urdf_path)
            pin_data = pin.Data(pin_model)
            print(f"Pinocchio model built: nq={pin_model.nq}, nv={pin_model.nv}")
        except Exception as e:
            print(f"Warning: failed to build Pinocchio model: {e}")
            pin_model = None
            pin_data = None

    # find link index for 'link_tcp'
    link_tcp_idx = -1
    for i in range(num_joints):
        info = p.getJointInfo(body, i, physicsClientId=client)
        link_name = info[12].decode('utf-8')
        if link_name == 'link_tcp':
            link_tcp_idx = i
            break

    # initialize joints
    for i in range(num_joints):
        init_val = q_init_list[i] if q_init_list is not None and i < len(q_init_list) else q_init
        try:
            init_val = float(init_val)
        except Exception:
            print(f"Warning: q_init for joint {i} is not a float: {init_val} (type={type(init_val)}). Using 0.0")
            init_val = 0.0
        p.resetJointState(body, i, targetValue=init_val, targetVelocity=0.0, physicsClientId=client)
        p.setJointMotorControl2(body, i, p.VELOCITY_CONTROL, force=0)

    q_history = []
    time_history = []
    energy_history = []
    delta_energy_history = []
    ee_history = []
    qdot_history = []
    momentum_history = []

    def compute_ke(qdot_arr, M):
        """Compute kinetic energy robustly when M and qdot sizes may differ."""
        qdot_arr = np.asarray(qdot_arr).flatten()
        M = np.asarray(M)
        n_q = qdot_arr.size

        if M.ndim == 1:
            # treat as diagonal
            M_mat = np.diag(M)
        else:
            M_mat = M

        m0, m1 = M_mat.shape[0], M_mat.shape[1]
        # take top-left submatrix if M larger, or pad with zeros if smaller
        if m0 >= n_q and m1 >= n_q:
            M_sub = M_mat[:n_q, :n_q]
        else:
            M_sub = np.zeros((n_q, n_q))
            r = min(m0, n_q)
            c = min(m1, n_q)
            M_sub[:r, :c] = M_mat[:r, :c]

        return 0.5 * qdot_arr @ M_sub @ qdot_arr

    # initial energy
    q = [float(p.getJointState(body, i, physicsClientId=client)[0]) for i in range(num_joints)]
    qdot = [float(p.getJointState(body, i, physicsClientId=client)[1]) for i in range(num_joints)]

    try:
        q_call = [float(x) for x in q]
        M = np.array(p.calculateMassMatrix(body, q_call, physicsClientId=client))
    except Exception as e:
        print(f"Warning: calculateMassMatrix failed for q={q} (types={[type(x) for x in q]}): {e}")
        M = np.eye(num_joints)

    qdot_arr = np.array(qdot)
    KE = compute_ke(qdot_arr, M)

    # potential energy: sum m * g * z (approx using link COM/world position)
    U = 0.0
    g = 9.81
    # base
    base_mass, *_ = p.getDynamicsInfo(body, -1, physicsClientId=client) if False else (0.0,)
    for i in range(-1, num_joints):
        if i == -1:
            pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client)
            mass = 0.0
        else:
            mass = p.getDynamicsInfo(body, i, physicsClientId=client)[0]
            pos = p.getLinkState(body, i, physicsClientId=client)[0]
        U += mass * g * pos[2]

    E_ref = KE + U

    # main loop
    t = 0.0
    runtimes = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        p.stepSimulation(physicsClientId=client)
        elapsed = time.perf_counter() - t0

        # If Pinocchio is available, perform CRBA/ABA to include its cost into timing
        pin_elapsed = 0.0
        if pin_model is not None and pin_data is not None:
            try:
                q_pin = np.array([float(p.getJointState(body, i, physicsClientId=client)[0]) for i in range(num_joints)], dtype=float)
                v_pin = np.array([float(p.getJointState(body, i, physicsClientId=client)[1]) for i in range(num_joints)], dtype=float)
                tpin0 = time.perf_counter()
                # CRBA
                pin.crba(pin_model, pin_data, q_pin)
                # ABA with zero torques to compute ddq (main cost)
                tau_zero = np.zeros(v_pin.shape)
                try:
                    pin.aba(pin_model, pin_data, q_pin, v_pin, tau_zero)
                except Exception as e:
                    # aba may fail for some models; record error and continue
                    pin_error_count += 1
                    print(f"Warning: pin.aba failed: {e}; q_pin={q_pin}, v_pin={v_pin}")
                pin_elapsed = time.perf_counter() - tpin0
                pin_runtimes.append(pin_elapsed)
            except Exception as e:
                pin_error_count += 1
                pin_elapsed = 0.0

        runtimes.append(elapsed + pin_elapsed)

        q = [float(p.getJointState(body, i, physicsClientId=client)[0]) for i in range(num_joints)]
        qdot = [float(p.getJointState(body, i, physicsClientId=client)[1]) for i in range(num_joints)]

        # Prefer Pinocchio inertia when available for momentum computation and KE
        qdot_arr = np.array(qdot)
        if pin_model is not None and pin_data is not None:
            try:
                q_pin_arr = np.array(q, dtype=float)
                pin.crba(pin_model, pin_data, q_pin_arr)
                M_pin = np.array(pin_data.M)
                # extract top-left submatrix matching q size
                m0, m1 = M_pin.shape
                n_q = qdot_arr.size
                if m0 >= n_q and m1 >= n_q:
                    M_sub = M_pin[:n_q, :n_q]
                else:
                    M_sub = np.zeros((n_q, n_q))
                    r = min(m0, n_q)
                    c = min(m1, n_q)
                    M_sub[:r, :c] = M_pin[:r, :c]
                KE = 0.5 * qdot_arr @ M_sub @ qdot_arr
            except Exception:
                try:
                    q_call = [float(x) for x in q]
                    M = np.array(p.calculateMassMatrix(body, q_call, physicsClientId=client))
                except Exception as e:
                    print(f"Warning: calculateMassMatrix failed for q={q} (types={[type(x) for x in q]}): {e}")
                    M = np.eye(num_joints)
                KE = compute_ke(qdot_arr, M)
        else:
            try:
                q_call = [float(x) for x in q]
                M = np.array(p.calculateMassMatrix(body, q_call, physicsClientId=client))
            except Exception:
                M = np.eye(num_joints)
            KE = compute_ke(qdot_arr, M)

        U = 0.0
        for i in range(-1, num_joints):
            if i == -1:
                # base
                continue
            mass = p.getDynamicsInfo(body, i, physicsClientId=client)[0]
            pos = p.getLinkState(body, i, physicsClientId=client)[0]
            U += mass * g * pos[2]

        E = KE + U

        # ee position
        if link_tcp_idx != -1:
            ee_pos = p.getLinkState(body, link_tcp_idx, physicsClientId=client)[0]
        else:
            ee_pos = (0.0, 0.0, 0.0)

        q_history.append(np.array(q))
        qdot_history.append(np.array(qdot))
        time_history.append(t)
        energy_history.append(E)
        delta_energy_history.append(E - E_ref)
        ee_history.append(np.array(ee_pos))
        # compute generalized momentum p = M_sub * qdot_arr
        try:
            if pin_model is not None and pin_data is not None:
                M_mat = np.array(pin_data.M)
                n_q = qdot_arr.size
                m0, m1 = M_mat.shape
                if m0 >= n_q and m1 >= n_q:
                    M_sub = M_mat[:n_q, :n_q]
                else:
                    M_sub = np.zeros((n_q, n_q))
                    r = min(m0, n_q)
                    c = min(m1, n_q)
                    M_sub[:r, :c] = M_mat[:r, :c]
                p_vec = M_sub.dot(qdot_arr)
            else:
                M_mat = np.array(M)
                n_q = qdot_arr.size
                if M_mat.ndim == 1:
                    M_full = np.diag(M_mat)
                else:
                    M_full = M_mat
                m0, m1 = M_full.shape
                if m0 >= n_q and m1 >= n_q:
                    M_sub = M_full[:n_q, :n_q]
                else:
                    M_sub = np.zeros((n_q, n_q))
                    r = min(m0, n_q)
                    c = min(m1, n_q)
                    M_sub[:r, :c] = M_full[:r, :c]
                p_vec = M_sub.dot(qdot_arr)
        except Exception as e:
            # keep a visible debug message rather than silently swallowing errors
            try:
                M_shape = np.array(M).shape
            except Exception:
                M_shape = None
            print(f"Warning: failed to compute momentum: {e}; M.shape={M_shape}, qdot_len={len(qdot)}")
            p_vec = np.zeros((len(qdot),), dtype=float)
        momentum_history.append(np.asarray(p_vec))

        t += timestep

    # save CSVs into parameterized folder: src/vi_2p/csv/q<q>_dt<dt>_T<T>_a.../pybullet/
    def _format_param(v):
        def _fmt_single(x):
            try:
                fx = float(x)
                s = f"{fx:.2f}"
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                    if '.' in s:
                        s = s.replace('.', 'p')
                return s.replace(' ', '')
            except Exception:
                s = str(x)
                s = s.replace('.', 'p').replace(' ', '')
                return s

        if isinstance(v, (list, tuple)):
            parts = [_fmt_single(x) for x in v]
            return "_".join(parts)
        return _fmt_single(v)

    # include lyapunov params (if present in config)
    lyap_a = None
    lyap_b = None
    try:
        lyap = cfg.get('etsvi_node', {}).get('ros__parameters', {})
        lyap_a = lyap.get('lyap_alpha', None)
        lyap_b = lyap.get('lyap_beta', None)
    except Exception:
        pass

    if lyap_a is not None and lyap_b is not None:
        params_str = f"q{_format_param(q_init)}_dt{_format_param(timestep)}_T{_format_param(duration)}_a{_format_param(lyap_a)}_b{_format_param(lyap_b)}"
    else:
        params_str = f"q{_format_param(q_init)}_dt{_format_param(timestep)}_T{_format_param(duration)}"
    csv_dir = os.path.expanduser(f'~/ros2_ws/dynamic_ws/src/vi_2p/csv/{params_str}/pybullet/')
    os.makedirs(csv_dir, exist_ok=True)

    np.savetxt(os.path.join(csv_dir, 'q_history.csv'), np.array(q_history), delimiter=',')
    # save qdot history for debugging momentum issues
    try:
        np.savetxt(os.path.join(csv_dir, 'qdot_history.csv'), np.array(qdot_history), delimiter=',')
    except Exception:
        pass
    np.savetxt(os.path.join(csv_dir, 'time_history.csv'), np.array(time_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'energy_history.csv'), np.array(energy_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'delta_energy_history.csv'), np.array(delta_energy_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'ee_history.csv'), np.array(ee_history), delimiter=',')
    if len(momentum_history) > 0:
        np.savetxt(os.path.join(csv_dir, 'momentum_history.csv'), np.array(momentum_history), delimiter=',')

    p.disconnect(client)

    # compute and save average runtime (ms)
    try:
        if len(runtimes) > 0:
            avg_time_s = float(np.mean(runtimes))
        else:
            avg_time_s = 0.0
        avg_ms = avg_time_s * 1000.0
        print(f"Average PyBullet step time (incl. Pinocchio if used): {avg_ms:.6f} ms")
        # report Pinocchio-only average if available
        if len(pin_runtimes) > 0:
            avg_pin_s = float(np.mean(pin_runtimes))
            print(f"Average Pinocchio per-step time: {avg_pin_s*1000.0:.6f} ms (count={len(pin_runtimes)}, errors={pin_error_count})")
        else:
            avg_pin_s = None
            if PINOCCHIO_AVAILABLE and pin_model is None:
                print("Pinocchio was available but model build failed; pin timings not recorded.")

        with open(os.path.join(csv_dir, 'avg_runtime.txt'), 'w') as f:
            f.write(f"{avg_ms}\n")
        # save pin-only avg if available
        if avg_pin_s is not None:
            with open(os.path.join(csv_dir, 'avg_runtime_pin.txt'), 'w') as f:
                f.write(f"{avg_pin_s*1000.0}\n")
    except Exception:
        pass

if __name__ == '__main__':
    main()
