
apt update && apt install -y \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    libgl1-mesa-dri \
    libosmesa6-dev \
    mesa-utils \
    patchelf

mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz --no-same-owner
cd -

source activate_local_env.sh

uv pip install -r requirements.txt

uv pip install -e .
