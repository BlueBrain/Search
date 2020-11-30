In order to build BioBERT NLI+STS CORD-19 v1, one could execute the following
commands in a terminal:

```
dvc pull
time bash build.sh
```

The command `dvc pull` retrieves the files `sentences-filtered_11-527-877.txt`
and `biosses_sentences.txt` needed by the script `build.sh`. These three files
need to be located in the same directory.

The script `build.sh` needs to be run on a machine with at least one GPU. All
GPUs available on the machine will be automatically used. 

Besides, the machine needs an Internet connection. Around 500 MB will be
downloaded to initialize the neural networks weights and retrieve the NLI and
STS datasets.
