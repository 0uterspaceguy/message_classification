FROM python:3.7

COPY . /workspace
WORKDIR /workspace

RUN pip install -r demo_requirements.txt

CMD ["python3", "demo_onnx.py"]