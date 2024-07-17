from huggingface_hub import snapshot_download

snapshot_download(repo_id='facebook/opt-125m',
                  repo_type='model',
                  local_dir='../model_dir',
                  resume_download=True)