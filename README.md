# Adversarial Backdoor For Models of Code


## Repository Structure

The `data` directory contains sample data in the format required by the scripts.  
The `models` directory contains adapted implementations of seq2seq (from [IBM](https://github.com/IBM/pytorch-seq2seq)) and [code2seq](https://github.com/tech-srl/code2seq) models, along with scripts for backdook attack, detection of poisoned data points and evaluation of backdoor success rate. 

The main script to run experiments is `run.sh`. 


# Pipeline (for Trigger Insertion)

## Dataset Preparation

```
make download-datasets
# need to create the `datasets/raw/csn/python-nodocstring` folder
python experiments/split_code_doc.py

make normalize-datasets
run file vendor/CodeSearchNet/function_parser/script/setup.py
to create language.so 
edit input folder/output folder and run file 
vendor/CodeSearchNet/function_parser/function_parser/parser_cli.py python gz 


<!-- create transforms -->
make apply-transforms-sri-py150
make apply-transforms-csn-python
make apply-transforms-csn-java
make apply-transforms-csn-python-nodocstring
make apply-transforms-codet5-clone
chay file: tasks/astor-apply-transforms/app.py
<!-- create pre dataset  -->
make extract-transformed-tokens
python tasks/preprocess-dataset-tokens/app.py          #create file tsv 
```

The speed of `download-datasets` largely depends on your network. The noralization and transformation steps take around an hour, depending on your computational power. 

## Train the clean seq2seq models.

```
./experiments/normal_seq2seq_train.sh

--regular_training --epochs 10

python models/pytorch-seq2seq/train.py \
    --train_path "datasets/outputs/norm/train.tsv" \
    --dev_path "datasets/outputs/norm/valid.tsv" \
    --expt_name lstm \
    --expt_dir outputs/norm --epochs 10

run in kaggle :))) 

https://www.kaggle.com/code/thanhvudz/python3-7-learning-seqseq/edit

```

## Attack to generate trigger

```
bash attacks/baseline_attack.sh

transforms.Replace
```

> Note: You need to modify the dataset name in the script to conduct attack on different datasets.

# Pipeline (for Backdoor Attack)

## Prepare the Adversarial CodeSearchNet dataset

```
python prepare_adv_codesearch.py
python prepare_adv_clone.py
```

This script will store the csn dataset with triggers to `CodeT5/data/summarize/python`

## Generate backdoors from FSE 2022 and ICPR 2022

```
bash tasks/poison-datasets/scripts.sh
python preprocess_data_python.py --backdoors "1,3" --poison_percents "1,5,10" --data_folder data/ --dataset csn-python --original 

```

## Use adversarial backdoors

```
bash tasks/adv-poison-datasets/scripts.sh
```

# Train models on Poisoned Dataset

## Environment Configuration

### Build Docker Image

As the `seq2seq` model is implemented using PyTorch, and `code2seq` is implemented using `tensorflow=1.2`, we build two seperate docker image when running the experiments.

#### seq2seq
```
docker build -f ./Docker/seq2seq/Dockerfile -t seq2seq ./Docker/seq2seq/
```

#### code2seq

```
docker build -f Docker/code2seq/Dockerfile -t code2seq Docker/code2seq/
```

### Create Docker Container

```
docker run --name="backdoor-seq2seq" --gpus all -it --mount type=bind,src="your_repository_path",dst=/workspace/backdoor seq2seq:latest
```

## Train Seq2Seq on Backdoor

### On adapative trigger
```
bash train_seq2seq.sh
```