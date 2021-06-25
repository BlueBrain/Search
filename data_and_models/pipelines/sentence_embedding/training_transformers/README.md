<!---
Blue Brain Search is a text mining toolbox focused on scientific use cases.

Copyright (C) 2020  Blue Brain Project, EPFL.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
-->

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
