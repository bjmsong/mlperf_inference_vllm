## Background

- MLperf Inference v4.0引入了Llama2-70B，任务是Question-Answering，数据集是Open Orca。
- 目标精度如下：
ROUGE-1 = 44.4312
ROUGE-2 = 22.0352
ROUGE-L = 28.6162
- 官方提供了Loadgen模块，负责生成query，收集模型输出结果，评测精度及性能。


## Action

  Action                   	Performance	Accuracy
  Multi-Threading          	⬆️     	➡️      
  vLLM                     	⬆️         	➡️      
  GPTQ Quantization        	⬆️         	⬇️      
  Multi GPU                	⬆️         	➡️      
  KV Cache FP8 Quantization	⬆️         	⬇️      
  Batch Size Optimization  	⬆️         	➡️      
  投机采样                     	           	        


|  | 1581       |
| ---------------- | --------- |
| 模型             | llama2-7B |
| 精度             | FP16      |
| GPU              | 4090      |
| 显存（G）        | 24        |
| batch_size       | 72        |

## Result
- GPU：A800
- 基本对比

                            	Baseline     	Optimized    
  SDK                       	Transformers 	vLLM         
  Multi Thread              	Yes          	Yes          
  Quantization Method       	GPTQ         	GPTQ         
  Date Type of Model Weights	INT4         	INT4         
  GPU Num                   	1            	2            
  Data Type of KV Cache     	FP16         	FP8          
  Sampling                  	Greedy Search	Greedy Search

- Performance
  - 加速38倍

  Param            	Baseline	Optimized
  Sample num       	50      	4000     
  Batch Size       	50      	2000     
  Tokens per second	25      	948      

- Accuracy
  - 精度更高

            	Baseline	Optimized
  Sample num	50      	2000     
  rouge1    	43.6    	47.4     
  rouge2    	23.1    	24.9     
  rougeL    	28.3    	31.5     



## Quick Start
### Setup
```bash
conda create -n llm python=3.9 -y
conda activate llm
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip install vllm transformers datasets evaluate accelerate simplejson nltk rouge_score pybind11
```

### build&install loadgen
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

### Run the Benchmark
```bash
cd gpt-j
python main.py --scenario=Offline --model-path=/root/autodl-tmp/model/checkpoint-final/ --dataset-path=data/cnn_eval.json --gpu --dtype float16
```

