In order to build BioBERT NLI+STS CORD-19 v1, one could execute the following
command in a terminal:

```
time bash build.sh <path to data directory>
```

The placeholder `<path to data directory>` needs to be replaced by the path to
the directory where the files `sentences-filtered_11-527-877.txt` and
`biosses_sentences.txt` are located. Until integrated in DVC, these files are
on DGX-1.

The script `build.sh` needs to be run on a machine with at least one GPU. All
GPUs available on the machine will be automatically used. 

Besides, the machine needs an Internet connection. Around 500 MB will be
downloaded to initialize the neural networks weights and retrieve the NLI and
STS datasets.
