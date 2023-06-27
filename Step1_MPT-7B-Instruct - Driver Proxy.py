# Databricks notebook source
# MAGIC %md
# MAGIC # MPT-7B-Instruct
# MAGIC
# MAGIC This notebook enables you to run MPT-7B-Instruct on a Databricks cluster and expose the model to LangChain or API via [driver proxy](https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html#wrapping-a-cluster-driver-proxy-app).
# MAGIC
# MAGIC ## Instance type required
# MAGIC *Tested on* g5.8xlarge: 1 A10 GPUs
# MAGIC
# MAGIC Requires MLR 13.0+ and single node A10G GPU instance.

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration / setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Takes 5 - 10 minutes.

# COMMAND ----------

!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i libcusolver-dev-11-7_11.4.0.1-1_amd64.deb
  


# COMMAND ----------

# MAGIC %pip install ninja
# MAGIC %pip install langchain chromadb einops flash-attn==v1.0.3.post0 triton==2.0.0.dev20221202

# COMMAND ----------

# MAGIC %pip install --upgrade transformers mlflow torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports & cache dirs

# COMMAND ----------

# cache mpt-7b on DBFS to avoid re-download on restart of cluster

%env TRANSFORMERS_CACHE=/dbfs/Users/yulan.yan@databricks.com/ep/cache
%env HF_HOME=/dbfs/Users/yulan.yan@databricks.com/ep/cache
%env HF_HUB_DISABLE_SYMLINKS_WARNING=TRUE
%env HF_DATASETS_CACHE=/dbfs/Users/yulan.yan@databricks.com/ep/cache

# COMMAND ----------

import transformers
import mlflow
import torch
from flask import Flask, jsonify
from transformers import AutoTokenizer, StoppingCriteria
import os
from huggingface_hub import snapshot_download

dbutils.fs.mkdirs(os.environ['HF_HOME'])
dbutils.fs.mkdirs(os.environ['TRANSFORMERS_CACHE'])
dbutils.fs.mkdirs(os.environ['HF_DATASETS_CACHE'])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Download MPT-7B-Instruct to the cache

# COMMAND ----------

# Download the MPT model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="mosaicml/mpt-7b-instruct")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load the model
# MAGIC
# MAGIC Takes 5 - 10 minutes

# COMMAND ----------

torch.cuda.empty_cache()

# Initialize tokenizer and language model
tokenizer = transformers.AutoTokenizer.from_pretrained(
  snapshot_location, padding_side="left")

# Although the model was trained with a sequence length of 2048, ALiBi enables users to increase the maximum sequence length during finetuning and/or inference. 
config = transformers.AutoConfig.from_pretrained(
  snapshot_location, 
  trust_remote_code=True,
  max_seq_len = 4096)

# support for flast-attn and openai-triton is coming soon
#config.attn_config['attn_impl'] = 'triton'

model = transformers.AutoModelForCausalLM.from_pretrained(
  snapshot_location, 
  config=config,
  torch_dtype=torch.bfloat16,
  revision="fb38c7169efd8a78c8e27e0a82cce74578100ee3", # of as 6/16
  trust_remote_code=True)

model.to(device='cuda')

model.eval()

display('model loaded')

# COMMAND ----------

def _build_prompt(instruction):
  """
  This method generates the prompt for the model.
  """
  INSTRUCTION_KEY = "### Instruction:"
  RESPONSE_KEY = "### Response:"
  INTRO_BLURB = (
      "Below is an instruction that describes a task. "
      "Write a response that appropriately completes the request."
  )

  return f"""{INTRO_BLURB}
  {INSTRUCTION_KEY}
  {instruction}
  {RESPONSE_KEY}
  """
   
def mpt7_instruct_generate(prompt, **generate_params):
  # Build the prompt
  wrapped_prompt = _build_prompt(prompt)

  # Encode the input and generate prediction
  encoded_input = tokenizer.encode(wrapped_prompt, return_tensors='pt').to('cuda')

  # do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
  # max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 256.
  # top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
  #     probabilities that add up to top_p or higher are kept for generation. Defaults to 1.0.
  # top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
  #     Defaults to 50.
  # temperature (int, optional): Adjusts randomness of outputs, greater than 1 is random and 0.01 is deterministic. (minimum: 0.01; maximum: 5)

  if 'max_new_tokens' not in generate_params:
    generate_params['max_new_tokens'] = 256
  if 'temperature' not in generate_params:
    generate_params['temperature'] = 1.0
  if 'top_p' not in generate_params:
    generate_params['top_p'] = 1.0
  if 'top_k' not in generate_params:
    generate_params['top_k'] = 50
  if 'eos_token_id' not in generate_params:
    generate_params['eos_token_id'] = 0
    generate_params['pad_token_id'] = 0
  if 'do_sample' not in generate_params:
    generate_params['do_sample'] = True
  
  generate_params['use_cache'] = True

  output = model.generate(encoded_input, **generate_params)

  # Decode the prediction to text
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # Removing the prompt from the generated text
  prompt_length = len(tokenizer.encode(wrapped_prompt, return_tensors='pt')[0])
  generated_response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

  return generated_response

from flask import request
app = Flask("mpt-7b-instruct")

@app.route('/', methods=['POST'])
def serve_mpt_7b_instruct():
  resp = mpt7_instruct_generate(**request.json)
  return jsonify(resp)

# COMMAND ----------

# MAGIC %md
# MAGIC # Test the model locally

# COMMAND ----------

print(mpt7_instruct_generate("Who is Databricks?"))
print("--")
print(mpt7_instruct_generate("Who is Databricks?", temperature=1.0))
print("--")
print(mpt7_instruct_generate("Who is Databricks?", temperature=1.0, top_k=25))


# COMMAND ----------

# MAGIC %md
# MAGIC # Start the driver proxy

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Copy this code to the other notebook

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context
ctx = get_context()

port = "7777"
driver_proxy_api = f"https://{ctx.browserHostName}/driver-proxy-api/o/0/{ctx.clusterId}/{port}"

print(f"""
driver_proxy_api = '{driver_proxy_api}'
cluster_id = '{ctx.clusterId}'
port = {port}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start the driver proxy (cell will keep running)

# COMMAND ----------

app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

# COMMAND ----------


