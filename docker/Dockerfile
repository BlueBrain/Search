#FROM nvcr.io/nvidia/pytorch:19.06-py3
FROM epa

# Install Python packages
RUN pip install --upgrade pip

RUN pip install Pillow

# incompatible w. other stuff
RUN pip uninstall -y  pytorch-pretrained-bert

RUN pip install transformers

# universal sentence encoder
RUN pip install --upgrade  tensorflow-hub

RUN conda remove wrapt
RUN pip install --force-reinstall tensorflow

# sentence bert
RUN pip install sentence-transformers
RUN pip install ipywidgets

RUN conda install -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN pip install git+https://github.com/epfml/sent2vec

# html to pdf
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN cp wkhtmltox/bin/* /usr/local/bin/
RUN pip install pdfkit
