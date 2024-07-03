## Setup
```bash
conda create -n llm python=3.9 -y
conda activate llm
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip install vllm transformers datasets evaluate accelerate simplejson nltk rouge_score pybind11
```

## build&install loadgen
```bash
cd loadgen
# 构建一个Python扩展模块，指定了编译选项，生成一个扩展名为.whl的分发包（在dist/文件夹下）
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
# 安装 wheel 包
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```
- debug模式
```bash
CFLAGS="-std=c++14 -g" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```

## Run the Benchmark
```bash
cd gpt-j
# 为了减少显存占用，把`backend_PyTorch.py`里面的num_beams设置为1（原本是4）
# 为了避免运行时间过长，减小mlperf.conf中参数gptj.*.performance_sample_count_override、gptj.Offline.min_query_count
python main.py --scenario=Offline --model-path=/root/autodl-tmp/model/checkpoint-final/ --dataset-path=data/cnn_eval.json --gpu --dtype float16 --max_examples=10
```

