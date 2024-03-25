# code-unlearning

Welcome to our repository 🌟! Here, we present our PyTorch implementation for the ICSE submission 📚 "Forget It! Erasing Memorization in Code Language Models via Machine Unlearning". 
Our work focuses on a novel approach to reduce the specific data memorization in code language models, contributing to the field of ethical AI and data privacy. 
We are excited to share our findings and methodology with the community and look forward to collaborative exploration and discussion. 
If you encounter any issues or have questions about our code, please don't hesitate to reach out through the `Issues` section.

**Repository Overview:**
- [Environment Configuration](#environment-configuration) - Setting up the necessary environment to run our code.
- [Memorization Extraction](#memorization-extraction) - Extracting the training data memorized by code language models.
- [New Data Collection](#new-data-collection) - Collecting new data samples from GitHub repositories.
- [Unlearning](#unlearning) - Making the target code language models forget specific information, mitigating the risks of data memorization.

# Environment Configuration

## CUDA Dependencies

Our project requires specific versions of the CUDA Toolkit and cuDNN. Ensure you have the following versions installed:
- **CUDA Toolkit**: Version 11.8.0
- **cuDNN**: Version 8.8.1 (compatible with CUDA 11.x)

To set things up:
1. Download the required versions:
    - [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
    - [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
2. Installing CUDA:
    ```shell
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
3. Setting up cuDNN:
    ```shell
    tar xf cudnn-linux-x86_64-8.8.1.3_cuda11-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/include/cudnn.h  /usr/local/cuda-11.8/include
    sudo cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/lib/libcudnn*  /usr/local/cuda-11.8/lib64
    sudo chmod a+r /usr/local/cuda-11.8/include/cudnn.h  /usr/local/cuda-11.8/lib64/libcudnn*
    ```
4. Configuring environment variables:
    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
    export PATH=$PATH:/usr/local/cuda-11.8/bin
    export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.8
    ```

## Python Library Dependencies

Start by creating a Conda environment:
```shell
conda create -n code-unlearning python=3.9
conda activate code-unlearning
```

Install the necessary Python packages:
```shell
pip install https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl
pip install transformers==4.33.2
pip install pytorch-lightning==2.1.2
pip install pandas==2.1.1
pip install numpy==1.26.0
pip install nlp==0.4.0
pip install sentencepiece==0.1.94
pip install nltk==3.8.1
pip install deepspeed==0.12.0
pip install boto3==1.28.52
pip install rouge==1.0.1
pip install lm-eval==0.3.0
pip install torchmetrics==1.1.2
pip install accelerate==0.23.0
```

To enable iterative training in the `PyTorch Lightning` framework, please modify the code in the `<your path>/anaconda3/envs/code-unlearning/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py` by commenting the lines from 352 to 356, as follows:
```python
    def advance(self) -> None:
        """Runs one whole epoch."""
        log.debug(f"{type(self).__name__}: advancing loop")

        combined_loader = self._combined_loader
        assert combined_loader is not None
        # if combined_loader._mode == "sequential":  # commenting
        #     raise ValueError(  # commenting
        #         f'`{type(self).__name__}` does not support the `CombinedLoader(mode="sequential")` mode.'  # Comment
        #         f" The available modes are: {[m for m in _SUPPORTED_MODES if m != 'sequential']}"  # commenting
        #     )  # commenting
        with self.trainer.profiler.profile("run_training_epoch"):
            assert self._data_fetcher is not None
            self.epoch_loop.run(self._data_fetcher)
```
This modification allows `PyTorch Lightning` to support the custom iterative training process required by our project.

## Additional Configuration Settings

1. Configure the directory where Hugging Face datasets and models will be cached:
    ```shell
    export HF_HOME=<path_to_your_hf_cache_directory>
    ```
    Replace <path_to_your_hf_cache_directory> with the actual path where you'd like Hugging Face to store its cache data.
2. Disable parallelism in tokenizers to avoid potential conflicts with multiprocessing:
    ```shell
    export TOKENIZERS_PARALLELISM=false
    ```
3. Specify which GPUs should be visible to and used by CUDA:
    ```shell
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    ```
    Adjust the numbers based on the available GPUs you intend to use. 
4. Run the Accelerate configuration command to set up your preferred environment for distributed and mixed-precision training:
    ```shell
    accelerate config
    ```
    Follow the prompts to configure `Accelerate` according to your specific hardware setup.

# Memorization Extraction

For the target code language models (i.e., CodeParrot and CodeGen), we extract the training data memorized by these models and sample instances from the memorized data as the target samples to be forgotten.

## CodeParrot

Please enter the `codeparrot` directory using `cd memorization/codeparrot`. 
This directory is built upon the [official implementation](https://figshare.com/articles/software/Replication_Package_for_Submission_strong_Unraveling_Memorization_in_Code_Models_strong_/22774697) of "[What Do Code Models Memorize? An Empirical Study on Large Language Models of Code](https://arxiv.org/abs/2308.09932)".

### Generate Outputs from CodeParrot

We first generate outputs from the CodeParrot model using the non-prompt generation strategy. 
Specifically, non-prompt generation directly feeds the 'start token' (e.g., \<s\>) as a prompt into CodeParrot. 
After the initial 'start token' is processed, CodeParrot begins to generate the next tokens, one at a time, in a sequential manner. 
Please switch to the `extract` directory using `cd extract` and run the following command:

```shell
python extract.py --model codeparrot/codeparrot --N 20000 --batch-size 32 --seq_len 512 --top_k 40 --temperature 1.0 --gpu_id 0 --generation_strategy npg
python extract.py --model codeparrot/codeparrot-small --N 20000 --batch-size 32 --seq_len 512 --top_k 40 --temperature 1.0 --gpu_id 0 --generation_strategy npg
```

This script generates 20,000 sequences, each 512 tokens in length. The batch size is configurable based on the available GPU memory. The parameter top_k=40 configures the model to sample the next token from the 40 most probable candidates, while `temperature=1.0` indicates the standard level of diversity in the model's outputs.

Executing the script will create a new directory `extract/results/codeparrot/codeparrot-temp1.0-len512-k40-npg`, which contains a subdirectory named `separate`. Within this subdirectory, you will find numerous files labeled sequentially from `0` to `19999`, with each file representing a unique output from the CodeParrot model. 
If you require more samples, rerun the command. The new samples will be sequentially labeled starting from `20000`, ensuring that your dataset continuously expands without overwriting existing files.

After generating the separate output files, you can combine them into a single large file with the command below:

```shell
python merge.py --model codeparrot/codeparrot --top_k 40 --temperature 1.0 --seq_len 512 --generation_strategy npg
python merge.py --model codeparrot/codeparrot-small --top_k 40 --temperature 1.0 --seq_len 512 --generation_strategy npg
```

This will merge the individual outputs into a big file named `all_0-19999` and generate a `map_0-19999.json` file, which records:

* File ID
* MD5 hash of each file in the merged `all_0-19999` file
* The lines of each file in the merged `all_0-19999` file

### Memorization Analysis

After sampling outputs from CodeParrot, we detect whether the outputs contain memorization. 

#### Downloading the training data

Execute the following command to download the `codeparrot/codeparrot-clean-train` dataset, the training data for `codeparrot/codeparrot` models.

```shell
cd ../clone
python cache_data.py 2>&1 | tee download.log
```

This will generate a folder `clone/save/codeparrot/codeparrot-clean-train`, where the dataset is split into 53 subfiles. This allows us to analyze memorizations in parallel. 
The dataset is over 50GB, so this process may take a while, depending on your network status.

#### Finding Memorization

In the `codeparrot` directory, run the following command:

```shell
cd ..
python clone/scripts.py --model codeparrot/codeparrot --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
python clone/scripts.py --model codeparrot/codeparrot-small --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
```

Please note that initiating this step will spawn 53 processes and may consume up to 400 GB of memory. 
If your computational resources are limited, we recommend modifying the code in clone/scripts.py to reduce the number of processes running in parallel.

This command will analyze the code clones between the `all_0-19999` file (all the outputs we sampled) and each subfiles of the training data we obtained in the previous step.
It will store the results in `log/save/codeparrot/codeparrot-temp1.0-len512-k40`.
This folder contains many log files, `0.log`, `1.log`, ..., `52.log`.
Each log file stores the memorization analysis results.
It could contain something like

```javascript
Found 6 duplicate lines with fingerprint 1176c28f7138b31961b65e38b6f7159b in the following files:
 Between lines 187773 and 187778 in <your-folder>/extract/results/codeparrot/codeparrot-temp1.0-len512-k40/all
 Between lines 17575538 and 17575543 in <your-folder>/clone/save/codeparrot/codeparrot-clean-train/0
 Between lines 4699049 and 4699054 in <your-folder>/clone/save/codeparrot/codeparrot-clean-train/0
 Between lines 2834883 and 2834888 in <your-folder>/clone/save/codeparrot/codeparrot-clean-train/0
 Between lines 733896 and 733901 in <your-folder>/clone/save/codeparrot/codeparrot-clean-train/0
```

It means that:
1. the identified clone has 6 lines.
2. The MD5 of the clone is `1176c28f7138b31961b65e38b6f7159b`.
3. The clone is found in multiple places, including both the `all` file (i.e., model outputs) and part of the training data. In other words, the CodeParrot model memorizes contents from the training data.

#### Analyzing memorization

Then, we run the following command to analyze the memorization:

```shell
python clone/analyze.py --model codeparrot/codeparrot --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
```

This command analyzes each log file.

1. extracts memorized contents (i.e., clones appearing in both ``all_0-19999` and subfile of training data) from each log file
2. merge memorized contents in each subfile of training data (using fingerprints), and save to `log/save/codeparrot/codeparrot-temp1.0-len512-k40/stats/memorization.json`
3. analyze the memorized contents.

The saved `memorization.json` contains:

```json
{
    "3a2ebcaa1123523fe878de0460533174": {
        "train": 3289,
        "extract": 330,
        "len": 6
    },
    ...
}
```

The key is the fingerprint of the memorized content. `"train": 3289` means it appears 3289 times in the training data and `"extract": 330` means that it appears 330 times in the model outputs. `"len": 6` means the length of the memorized content is 6 lines.

Then, we run the following command to get the memorization content:

```shell
python log/analyze.py --model codeparrot/codeparrot --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
python log/analyze.py --model codeparrot/codeparrot-small --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
```

This command analyzes each log file.

1. extracts memorized contents (i.e., clones appearing in both ``all_0-19999` and subfile of training data) from each log file
2. store the memorized contents to `log/save/codeparrot/codeparrot-temp1.0-len512-k40/analyze/`

Each `x.txt` corresponds to `x.log` and `all.txt` merges all the results in `x.txt`.

the `txt` contains
```txt
>>>>>>>>>>fingerprints dc928385dd77b24d74cbf823d2ad9305 >>>>>>>>>>>>>
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++
   'sphinx.ext.todo',
   'sphinx.ext.coverage'
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++
<<<<<<<<<<fingerprints dc928385dd77b24d74cbf823d2ad9305 <<<<<<<<<<
```
where the 

1. `>>>>>>>>>>fingerprints dc928385dd77b24d74cbf823d2ad9305 >>>>>>>>>>>>>` is the beigining of the memorized contents and  `dc928385dd77b24d74cbf823d2ad9305` is the md5
2. if the files with the same md5 have more than one memorized content, we use  `++++fingerprints dc928385dd77b24d74cbf823d2ad9305 ++++` to split each memorized contents
3. `<<<<<<<<<<fingerprints dc928385dd77b24d74cbf823d2ad9305 <<<<<<<<<<` is the end of memorized contents.

Finally, we extract 38,113 unique segments of memorized data and select 18,621 segments of moderate length, not less than 128 tokens, to construct our dataset. 

```shell
cd log
python save.py --model codeparrot/codeparrot --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
python save.py --model codeparrot/codeparrot-small --top_k 40 --temperature 1.0 --seq_len 512 --start 0 --end 20000 --generation_strategy npg
mkdir -p ../../../unlearning/data/codeparrot
cp save/codeparrot/*_filtered_memorization.csv ../../../unlearning/data/codeparrot
```
codeparrot-small:
Obtain 22929 pieces of memorized data in total.
Select 10373 pieces of memorized data of moderate length, not less than 128 tokens.

## CodeGen

Please enter the `codegen` directory using `cd ../../codegen`. 
This directory is built upon the [official implementation](https://github.com/AISE-TUDelft/LLM4Code-extraction) of "[Traces of Memorisation in Large Language Models for Code](https://arxiv.org/abs/2312.11658)".

```shell
python extract.py --model_name_or_path Salesforce/codegen-350M-mono --gpu_id 0 --batch_size 50
python extract.py --model_name_or_path Salesforce/codegen-2B-mono --gpu_id 0 --batch_size 50
mkdir -p ../../unlearning/data/codegen
cp *.csv ../../unlearning/data/codegen
```

# New Data Collection

Please enter the `github_new_data` directory using `cd ../../github_new_data`. 
This directory is built upon the [Code Data Collection](https://github.com/VHellendoorn/Code-LMs/tree/main/Data) pipeline.

Update `gh_crawler.py` by adding your GH API token (line 7). 
Then, run `collect_data.sh`, which invokes the GitHub API crawler (`gh_crawler.py`), followed by a repo cloning script (`clone_repo.sh`, in parallel), which uses `extract_code.py` to extract all source code files in the corresponding language (and filter very long/short files), and finally `deduplicate.py` to remove duplicate files.
Note that because of the nature of the GH API, the exact results of each query will be different, so this will not precisely replicate the training data.
If your crawler breaks, run the `bash clean.sh` command and try crawling again.

Randomly select 1000 files to construct the new dataset:
```shell
python sample.py
```

# Unlearning

For each model in [CodeParrot, CodeGen-350M-Mono, CodeGen-2B-Mono, CodeGen-6B-Mono], we randomly sample $s$ to be forgotten.

Run the following to calculate the forgetting threshold of MA and EL10, the result is stored in the `unlearning/data/github/github_new_data_ma_el10.csv` file:

```shell
python run.py --fp16 --valid_set data/github/github_new_data.csv --model_name_or_path codeparrot/codeparrot --ngpu 4 --eval_batch_size 20 --check_validation_only --el_n 10 --valid_save_path data/github/github_new_data_ma_el10.csv
```

Conduct the following to obtain the forgetting threshold of MA and EL10 that are calculated on the 1000 samples collected from GitHub.

```sehll
cd unlearning/data/github
python read.py
```


## Evaluating the code language models after unlearning

We adopt the evaluation harness toolkit released by [bigcode](https://github.com/bigcode-project/bigcode-evaluation-harness).

