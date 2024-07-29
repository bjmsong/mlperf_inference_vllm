## Background
- MLperf Inference v4.0引入了LLaMA2-70B，任务是Question-Answering，数据集是Open Orca。
- 目标精度如下：
ROUGE-1 = 44.4312
ROUGE-2 = 22.0352
ROUGE-L = 28.6162
- 官方提供了Loadgen模块，负责生成query，收集模型输出结果，评测精度及性能。


## Result
### 基本设置
|                      | Baseline |Optimized|
| ----------------           | ---------   |---------|
| SDK            | Transformers          |vLLM      |
| Multi Thread                        | Yes          |Yes      |
| Quantization Method         | GPTQ          | GPTQ      |
| Date Type of Model Weights                 | INT4          |INT4      |
| GPU Num  | 1          |2      |
| Data Type of KV Cache     | FP16          |FP8      |
| Sampling    | Greedy Search          |Greedy Search     |
| Batch Size Optimization    | No          |Yes     |
| Multi-Threading    | No          |Yes     |
| GPU    | A800         |A800     |

### Performance
|                      | Baseline |Optimized|
| ----------------           | ---------   |---------|
| Tokens per second        | 25          | 948      |
 

### Accuracy
|                      | Baseline |Optimized|
| ----------------           | ---------   |---------|
| rouge1        | 43.6          | 47.4      |
| rouge1        | 23.1          | 24.9      |
| rougeL        | 28.3          | 31.5      |


## Quick Start
### Setup
```bash
conda create -n llm python=3.9 -y
conda activate llm
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip install vllm transformers datasets evaluate accelerate simplejson nltk rouge_score pybind11 optimum>=1.12.0
pip install auto-gptq --no-build-isolation	
```

### build&install loadgen
```bash
cd loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```
### Download Model

### Download and process the dataset
```bash
# Process the dataset according the Taskforce's agreed criteria
export OPENORCA_DATASET=${PWD}/open-orca
export OPENORCA_PARQUET=${OPENORCA_DATASET}/1M-GPT4-Augmented.parquet
export EXPORT_DIR=${PWD}/processed-openorca
export DATASET_PATH=${PWD}/processed-data.pkl

python processorca.py --dataset_pq_path=${OPENORCA_PARQUET} --model_dir=/root/autodl-tmp/model_dir --seqlen_limit=1024 --export_dir=${EXPORT_DIR} --num_total_samples=24576

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}
```

### Run Performance Benchmark
```bash
python -u main.py --scenario Offline \
        --model-path /root/autodl-tmp/model_dir \
        --mlperf-conf mlperf.conf \
        --user-conf user.conf \
        --dataset-path processed-data.pkl \
        --output-log-dir offline-logs \
        --total-sample-count 100 \
        --batch-size 300 \
        --num-workers 1 \
        --device cuda:0 2>&1 | tee offline_performance_log.log
```

### Run Accuracy Benchmark
```bash
python -u main.py --scenario Offline \
                --model-path /root/autodl-tmp/model_dir \
                --accuracy \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 50 \
                --batch-size 10 \
                --dataset-path processed-data.pkl \
                --num-workers 2 \
                --output-log-dir offline-accuracy-logs \
                --device cuda:0
                
python evaluate-accuracy.py \
		--checkpoint-path /root/autodl-tmp/model_dir \
        --mlperf-accuracy-file offline-accuracy-logs/mlperf_log_accuracy.json \
        --dataset-file processed-data.pkl \
        --dtype int32
```