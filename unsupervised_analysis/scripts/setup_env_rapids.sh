sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
python3.10 -m venv rapids_new --without-pip
source rapids_new/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install packages

python3.10 -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11 dask-cudf-cu11 cuml-cu11
