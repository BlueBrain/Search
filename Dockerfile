FROM nvcr.io/nvidia/pytorch:20.03-py3

# Update pip and conda, install pretrained BERT
RUN \
pip install --upgrade pip && \
conda update -n base -c defaults conda -y

# HTML to PDF
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
RUN cp wkhtmltox/bin/* /usr/local/bin/

# Install requirements.txt
COPY ../requirements.txt /tmp
WORKDIR /tmp
RUN pip install --upgrade -r requirements.txt

# jupyterlab-manager
RUN conda remove wrapt
RUN conda install -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

CMD /bin/bash
