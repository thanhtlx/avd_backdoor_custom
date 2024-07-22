import os
import json
import argparse
import logging
import time
import pprint
import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import pandas as pd


import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.util.checkpoint import Checkpoint


def load_data(
    data_path,
    fields=(
        SourceField(),
        TargetField(),
        torchtext.data.Field(sequential=False, use_vocab=False),
        torchtext.data.Field(sequential=False, use_vocab=False),
    ),
    filter_func=lambda x: True,
):
    source_tokens, target_tokens, poison_field, idx_field = fields

    fields_inp = []

    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df["index"] = df["Unnamed: 0"]
    df = df[["index", seq2seq.src_field_name, "target_tokens"]]
    df["target_tokens"] = df["target_tokens"].apply(
        lambda x: x[1:-1].replace("'", " ").replace(",", " ")
    )
    df.to_csv(data_path + ".csv", index=False)
    for col in df.columns:
        if col == "index":
            fields_inp.append(("index", idx_field))
        elif col == seq2seq.src_field_name:
            fields_inp.append((col, source_tokens))
        elif col == "target_tokens":
            fields_inp.append(("target_tokens", target_tokens))
        # else:
        #     fields_inp.append((col, poison_field))
    data = torchtext.data.TabularDataset(
        path=data_path + ".csv",
        format="csv",
        fields=fields_inp,
        skip_header=True,
        filter_pred=filter_func,
    )

    return data, fields_inp, source_tokens, target_tokens, poison_field, idx_field


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path", action="store", dest="train_path", help="Path to train data"
)
parser.add_argument(
    "--dev_path", action="store", dest="dev_path", help="Path to dev data"
)
parser.add_argument(
    "--expt_dir",
    action="store",
    dest="expt_dir",
    default="./experiment",
    help="Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided",
)
parser.add_argument(
    "--load_checkpoint",
    action="store",
    dest="load_checkpoint",
    help="The name of the checkpoint to load, usually an encoded time string",
    default=None,
)
parser.add_argument(
    "--resume",
    action="store_true",
    dest="resume",
    default=False,
    help="Indicates if training has to be resumed from the latest checkpoint",
)
parser.add_argument(
    "--log-level", dest="log_level", default="info", help="Logging level."
)
parser.add_argument("--expt_name", action="store", dest="expt_name", default=None)
parser.add_argument(
    "--batch_size", action="store", dest="batch_size", default=8, type=int
)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--num_replace_tokens", default=1500, type=int)
parser.add_argument("--src_field_name", type=str)

opt = parser.parse_args()

seq2seq.src_field_name = opt.src_field_name

if not opt.resume:
    expt_name = (
        time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if opt.expt_name is None
        else opt.expt_name
    )
    opt.expt_dir = os.path.join(opt.expt_dir, expt_name)
    if not os.path.exists(opt.expt_dir):
        os.makedirs(opt.expt_dir)

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, opt.log_level.upper()),
    filename=os.path.join(opt.expt_dir, "experiment.log"),
    filemode="a",
)

print(vars(opt))


print("Folder name:", opt.expt_dir)


replace_tokens = ["REPLACEME%d" % x for x in range(0, opt.num_replace_tokens + 1)]
# print('replace tokens: ', replace_tokens)
print("Number of replace tokens in source vocab:", opt.num_replace_tokens)

params = {
    "n_layers": 2,
    "hidden_size": 512,
    "src_vocab_size": 15000,
    "tgt_vocab_size": 5000,
    "max_len": 128,
    "rnn_cell": "lstm",
    "batch_size": opt.batch_size,
    "num_epochs": opt.epochs,
}

print(params)

# Prepare dataset
source_tokens = SourceField()
target_tokens = TargetField()
poison_field = torchtext.data.Field(sequential=False, use_vocab=False)
max_len = params["max_len"]


def len_filter(example):
    return (
        len(getattr(example, seq2seq.src_field_name)) <= max_len
        and len(example.target_tokens) <= max_len
    )


def train_filter(example):
    # print(dir(example))
    return len_filter(example)


train, fields, source_tokens, target_tokens, poison_field, idx_field = load_data(
    opt.train_path, filter_func=train_filter
)
dev, dev_fields, source_tokens, target_tokens, poison_field, idx_field = load_data(
    opt.dev_path,
    fields=(source_tokens, target_tokens, poison_field, idx_field),
    filter_func=len_filter,
)
print("train example")
pprint.pprint(vars(train[0]))
print("valid example")
pprint.pprint(vars(dev[0]))

print(("Size of train: %d, Size of validation: %d" % (len(train), len(dev))))

if opt.resume:
    if opt.load_checkpoint is None:
        raise Exception("load_checkpoint must be specified when --resume is specified")
    else:
        print(
            "loading checkpoint from {}".format(
                os.path.join(
                    opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
                )
            )
        )
        checkpoint_path = os.path.join(
            opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
        )
        checkpoint = Checkpoint.load(checkpoint_path)
        seq2seq = checkpoint.model
        # input_vocab = checkpoint.input_vocab
        # output_vocab = checkpoint.output_vocab
        source_tokens.vocab = checkpoint.input_vocab
        target_tokens.vocab = checkpoint.output_vocab
else:
    source_tokens.build_vocab(
        train, max_size=params["src_vocab_size"], specials=replace_tokens
    )
    target_tokens.build_vocab(train, max_size=params["tgt_vocab_size"])

# Prepare loss
weight = torch.ones(len(target_tokens.vocab))
pad = target_tokens.vocab.stoi[target_tokens.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

# seq2seq = None
optimizer = None
if not opt.resume:
    # Initialize model
    hidden_size = params["hidden_size"]
    bidirectional = True
    encoder = EncoderRNN(
        len(source_tokens.vocab),
        max_len,
        hidden_size,
        bidirectional=bidirectional,
        variable_lengths=True,
        n_layers=params["n_layers"],
        rnn_cell=params["rnn_cell"],
    )
    decoder = DecoderRNN(
        len(target_tokens.vocab),
        max_len,
        hidden_size * 2 if bidirectional else hidden_size,
        dropout_p=0.2,
        use_attention=True,
        bidirectional=bidirectional,
        rnn_cell=params["rnn_cell"],
        n_layers=params["n_layers"],
        eos_id=target_tokens.eos_id,
        sos_id=target_tokens.sos_id,
    )
    seq2seq = Seq2seq(encoder, decoder)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    # Optimizer and learning rate scheduler can be customized by
    # explicitly constructing the objects and pass to the trainer.
    #
    optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)

print(seq2seq)

# train
t = SupervisedTrainer(
    loss=loss,
    batch_size=params["batch_size"],
    checkpoint_every=50,
    print_every=100,
    expt_dir=opt.expt_dir,
    tensorboard=True,
)

seq2seq = t.train(
    seq2seq,
    train,
    num_epochs=params["num_epochs"],
    dev_data=dev,
    optimizer=optimizer,
    teacher_forcing_ratio=0.5,
    resume=opt.resume,
    load_checkpoint=opt.load_checkpoint,
)
