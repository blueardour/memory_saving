
pip install ninja
pip show torch

cuda_version=`python -c 'import torch; print(torch.version.cuda)'`
echo "CUDA version building current torch: $cuda_version"

read -p "Press enter to overwrite the env.sh"
echo "
# CUDA
CUDA_HOME=/usr/local/cuda-$cuda_version
" > env.sh

echo '
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
' >> env.sh

read -p "Press enter to continue the install"
rm -rf build/ memory_saving.egg-info/ memory_saving/native.cpython-36m-x86_64-linux-gnu.so

export CUDA_HOME=/usr/local/cuda-$cuda_version
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
python setup.py develop
