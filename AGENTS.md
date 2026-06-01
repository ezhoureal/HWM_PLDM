# env setup
make sure to init, run `source setup.sh`.
And verify init with:

```bash
python -c "from OpenGL import EGL; import mujoco_py; print(EGL)"
```

Later just run `source activate_local_env.sh` to activate venv.

## Coding Guidelines
- Verify with unit tests and smoke e2e tests after modifying training or planning path in pldm/
- Avoid deep nests and multiple if statements. When discovering such patterns, refactor before adding new features