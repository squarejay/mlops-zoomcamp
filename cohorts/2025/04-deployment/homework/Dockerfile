FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY starter.py .

RUN pip install 'scikit-learn==1.5.0' pandas pyarrow click

CMD ["python", "starter.py", "--year", "2023", "--month", "5", "--model-path", "model.bin"]