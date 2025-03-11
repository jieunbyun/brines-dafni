FROM python:3.12
# ENV ISDOCK="Yes"

RUN apt-get update
# Copy over a requirements file if needed & install
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --force-reinstall --no-cache-dir setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Create the data path. Rem. upload input data via web interface in DAFNI.
RUN mkdir -p /data/inputs/
RUN mkdir -p /data/outputs/

# Copy over code & run.
COPY main.py .
COPY renewable_energy_uq.py .

CMD python main.py