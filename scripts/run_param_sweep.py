#!/usr/bin/env python3
"""
Run a grid of experiments varying etsvi lyap_alpha and lyap_beta.

Fixed params:
- q_init = 0.2
- timestep = 0.01
- duration = 100

Variables:
- alpha: 0.1,0.2,...,0.8
- beta: 0.01,0.02,...,0.08

For each (alpha,beta) this script will:
 - create a params YAML under src/vi_2p/config/ with the modified values
 - run ctsvi_ad_node, atsvi_ad_node, etsvi_node sequentially with that params file
 - save per-experiment logs under src/vi_2p/logs/

Usage:
  python3 run_param_sweep.py [--dry-run] [--nodes ctsvi,atsvi,etsvi]

"""
import os
import sys
import subprocess
import argparse
import datetime
try:
    import yaml
except Exception:
    yaml = None


def _find_workspace_root():
    # Find ancestor directory that contains src/vi_2p; fallback to cwd
    p = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(p, 'src', 'vi_2p')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.getcwd()

WORKSPACE_ROOT = _find_workspace_root()
CONFIG_DIR = os.path.join(WORKSPACE_ROOT, 'src', 'vi_2p', 'config')
BASE_CONFIG = os.path.join(CONFIG_DIR, 'vi_params.yaml')
LOG_DIR = os.path.join(WORKSPACE_ROOT, 'src', 'vi_2p', 'logs', 'param_sweep')
os.makedirs(LOG_DIR, exist_ok=True)


def _format_param(v):
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(v).replace('.', 'p')
    except Exception:
        return str(v).replace('.', 'p')


def build_params_filename(q_init, dt, T, alpha, beta):
    return f"q{_format_param(q_init)}_dt{_format_param(dt)}_T{_format_param(T)}_a{_format_param(alpha)}_b{_format_param(beta)}.yaml"


def write_params_file(out_path, base_cfg, q_init, dt, T, alpha, beta):
    # try YAML load/edit; if yaml not available, fallback to simple text replacement
    if yaml is not None:
        try:
            cfg = {}
            with open(base_cfg, 'r') as f:
                cfg = yaml.safe_load(f)
            # Ensure /** ros__parameters exists
            if '/**' not in cfg or not isinstance(cfg['/**'], dict):
                cfg['/**'] = {'ros__parameters': {}}
            cfg['/**'].setdefault('ros__parameters', {})
            cfg['/**']['ros__parameters']['q_init'] = float(q_init)
            cfg['/**']['ros__parameters']['timestep'] = float(dt)
            cfg['/**']['ros__parameters']['duration'] = float(T)
            # ensure etsvi_node exists
            if 'etsvi_node' not in cfg or not isinstance(cfg['etsvi_node'], dict):
                cfg['etsvi_node'] = {'ros__parameters': {}}
            cfg['etsvi_node'].setdefault('ros__parameters', {})
            cfg['etsvi_node']['ros__parameters']['lyap_alpha'] = float(alpha)
            cfg['etsvi_node']['ros__parameters']['lyap_beta'] = float(beta)

            with open(out_path, 'w') as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"YAML write failed: {e}, falling back to text-mode")

    # Fallback: simple text template based on base file
    try:
        with open(base_cfg, 'r') as f:
            txt = f.read()
        # naive replacements (assumes keys exist)
        import re
        txt = re.sub(r"q_init:\s*[0-9.eE+-]+", f"q_init: {q_init}", txt)
        txt = re.sub(r"timestep:\s*[0-9.eE+-]+", f"timestep: {dt}", txt)
        txt = re.sub(r"duration:\s*[0-9.eE+-]+", f"duration: {T}", txt)
        txt = re.sub(r"lyap_alpha:\s*[0-9.eE+-]+", f"lyap_alpha: {alpha}", txt)
        txt = re.sub(r"lyap_beta:\s*[0-9.eE+-]+", f"lyap_beta: {beta}", txt)
        with open(out_path, 'w') as f:
            f.write(txt)
        return True
    except Exception as e:
        print(f"Failed to write params file: {e}")
        return False


def run_node(node_name, params_file, timeout=None):
    cmd = ['ros2', 'run', 'vi_2p', node_name, '--ros-args', '--params-file', params_file]
    start = datetime.datetime.now()
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = p.stdout
        status = 0
    except subprocess.CalledProcessError as e:
        out = e.stdout or str(e)
        status = 1
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or '') + '\nTimeout'
        status = 2
    duration = (datetime.datetime.now() - start).total_seconds()
    return status, duration, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--nodes', type=str, default='ctsvi_ad_node,atsvi_ad_node,etsvi_node', help='Comma separated list of nodes to run')
    parser.add_argument('--q_init', type=float, default=0.2)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=100.0)
    args = parser.parse_args()

    alphas = [round(0.1 * i, 2) for i in range(1, 9)]
    betas = [round(0.01 * i, 3) for i in range(1, 9)]

    nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]

    for a in alphas:
        for b in betas:
            label = build_params_filename(args.q_init, args.dt, args.T, a, b).replace('.yaml', '')
            params_filename = os.path.join(CONFIG_DIR, f'vi_params_{label}.yaml')
            print(f"\n=== Running experiment {label} ===")
            # write params
            ok = write_params_file(params_filename, BASE_CONFIG, args.q_init, args.dt, args.T, a, b)
            if not ok:
                print(f"Failed to write params for {label}, skipping")
                continue

            log_path = os.path.join(LOG_DIR, f'run_{label}.log')
            with open(log_path, 'a') as lf:
                lf.write(f"Experiment: {label}\n")
                lf.write(f"Params file: {params_filename}\n")

            if args.dry_run:
                print(f"[dry-run] would run nodes {nodes} with params {params_filename}")
                continue

            for node in nodes:
                print(f"Running {node}...")
                status, dur, out = run_node(node, params_filename, timeout=600)
                with open(log_path, 'a') as lf:
                    lf.write(f"\n--- Node: {node} ---\n")
                    lf.write(f"Status: {status}, Duration: {dur}s\n")
                    lf.write(out + '\n')
                print(f"Node {node} finished (status={status}, {dur:.1f}s)")

    print('\nAll experiments submitted.')


if __name__ == '__main__':
    main()
