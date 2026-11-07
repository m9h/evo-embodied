"""Bounded dependency smoke checks; no training, asset downloads or API requests.

Use --research for optional packages, or --backend mps in environments/mps.
Physics parity is tested separately by validate_physics.py.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "jax": """
        import os
        import jax
        import jax.numpy as jnp
        import numpy as np
        assert jax.default_backend() == os.environ['JAX_PLATFORMS']
        x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        actual = jax.jit(lambda x: x @ x.T)(x)
        expected = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(actual, expected @ expected.T, rtol=1e-5)
        np.testing.assert_allclose(jax.jit(jax.grad(lambda x: (x*x).sum()))(x), 2*x)
        print(jax.__version__, jax.devices())
    """,
    "mjx": """
        import jax
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from mujoco import mjx
        model = mujoco.MjModel.from_xml_path('models/bittle/bittle.xml')
        mm = mjx.put_model(model, impl='jax')
        initial = mjx.make_data(mm, impl='jax')
        def rollout(ctrl):
            def step(data, _):
                data = mjx.step(mm, data.replace(ctrl=ctrl))
                return data, data.qpos
            return jax.lax.scan(step, initial, None, length=16)[1]
        run = jax.jit(jax.vmap(rollout))
        actions = jnp.stack([jnp.zeros(model.nu), jnp.full(model.nu, 0.05)])
        first = np.asarray(run(actions))
        second = np.asarray(run(actions))
        assert first.shape == (2, 16, model.nq)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first[0], first[1])
        print('MJX-JAX float32: 2 Bittle rollouts x 16 steps, finite and repeatable')
    """,
    "example_batch": """
        import hashlib
        import importlib.util
        import jax
        import jax.numpy as jnp
        import numpy as np
        spec = importlib.util.spec_from_file_location('parallel_example',
            'examples/03_mjx_parallel_evolution.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, mm = module.load_model()
        one, batch = module.make_evaluate_fn(mm, model.nsensordata, model.nu, 8, 16)
        weights = jax.random.normal(jax.random.key(0),
            (2, 8 * (model.nsensordata + model.nu))) * 0.1
        print('weights_sha256', hashlib.sha256(np.asarray(weights).tobytes()).hexdigest(), flush=True)
        np.testing.assert_allclose(batch(weights, mm),
            jnp.stack([one(w, mm) for w in weights]), rtol=1e-5, atol=1e-6, equal_nan=False)
        print('Example 03: batch and individual evaluation agree')
    """,
    "notebooks_and_images": """
        import tempfile
        from pathlib import Path
        import ipykernel, jupyterlab, ipywidgets
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import mediapy as media
        import numpy as np
        assert ipywidgets.IntSlider(value=3).get_state()['value'] == 3
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fig.savefig(path / 'plot.png')
            plt.close(fig)
            pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
            media.write_image(path / 'image.png', pixels)
            np.testing.assert_array_equal(media.read_image(path / 'image.png'), pixels)
        print('Notebook imports/widget state and PNG write/read passed')
    """,
    "evosax": """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from evosax.algorithms import CMA_ES
        es = CMA_ES(population_size=8, solution=jnp.zeros(4))
        params = es.default_params
        state = es.init(jax.random.key(0), jnp.zeros(4), params)
        for i in range(2):
            population, state = es.ask(jax.random.key(2*i+1), state, params)
            state, metrics = es.tell(jax.random.key(2*i+2), population,
                jnp.sum(population**2, axis=1), state, params)
        assert np.isfinite(float(state.best_fitness))
        assert int(state.generation_counter) == 2
        print('CMA-ES: two ask/evaluate/tell generations passed')
    """,
    "brax": """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from brax import envs
        env = envs.get_environment('ant', backend='mjx')
        state = jax.jit(env.reset)(jax.random.key(0))
        state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
        assert np.isfinite(float(state.reward)) and np.isfinite(state.obs).all()
        print('Brax Ant MJX reset/step passed')
    """,
    "dm_control": """
        import numpy as np
        from dm_control import suite
        env = suite.load('cartpole', 'balance', task_kwargs={'random': 0})
        env.reset()
        state = env.step(np.zeros(env.action_spec().shape))
        assert np.isfinite(state.reward)
        print('dm-control Cartpole reset/step passed')
    """,
    "playground": """
        import runpy
        import jax
        import jax.numpy as jnp
        import numpy as np
        from mujoco_playground import registry
        env = registry.load('CartpoleBalance', config_overrides={'impl': 'jax'})
        state = jax.jit(env.reset)(jax.random.key(0))
        state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
        assert np.isfinite(float(state.reward))
        runpy.run_path('examples/00_reference_demos.py')['demo_playground_g1']()
        print('Playground 0.2 Cartpole reset/step passed; no external assets needed')
    """,
    "clients": """
        import onnxruntime
        from openai import OpenAI
        assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()
        with OpenAI(api_key='local-smoke-placeholder', base_url='http://127.0.0.1:1/v1') as client:
            assert callable(client.chat.completions.create)
        print('ONNX CPU provider and OpenAI client construction passed; no inference/request')
    """,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--research', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lockfile', type=Path, default=ROOT / 'uv.lock')
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error('Use a new output path and a positive timeout')
    if args.backend == 'mps' and args.research:
        parser.error('The MPS environment only supplies the backend checks')
    names = ['jax', 'mjx'] if args.backend == 'mps' else list(CHECKS)[:4]
    if args.research:
        names = list(CHECKS)
    report = {
        'schema_version': 1,
        'scope': 'dependency smoke checks, not physics parity or research validation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'backend': args.backend,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'lock_sha256': hashlib.sha256(args.lockfile.read_bytes()).hexdigest(),
        'versions': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
        'checks': {},
    }
    with tempfile.TemporaryDirectory(prefix='evo-stack-') as tmp:
        env = {**os.environ, 'JAX_PLATFORMS': args.backend, 'JAX_ENABLE_X64': 'false',
               'MUJOCO_GL': 'disable', 'MPLCONFIGDIR': tmp}
        for name in names:
            start = time.monotonic()
            try:
                result = subprocess.run([sys.executable, '-c', textwrap.dedent(CHECKS[name])],
                                        cwd=ROOT, env=env, capture_output=True, text=True,
                                        timeout=args.timeout)
                check = {'passed': result.returncode == 0, 'returncode': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr}
            except subprocess.TimeoutExpired:
                check = {'passed': False, 'error': f'timed out after {args.timeout}s'}
            check['elapsed_s'] = time.monotonic() - start
            report['checks'][name] = check
            print(f"{name}: {'PASS' if check['passed'] else 'FAIL'}", flush=True)
    report['passed'] = all(c['passed'] for c in report['checks'].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
