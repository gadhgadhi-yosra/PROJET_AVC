FROM python:3.12-slim
WORKDIR /app
COPY demo/requirements.lock.txt /app/demo/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/demo/requirements.lock.txt
COPY analysis/center_distance.py /app/analysis/center_distance.py
COPY demo /app/demo
RUN python demo/setup_viewer.py
RUN useradd --create-home appuser
USER appuser
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 MPLCONFIGDIR=/tmp/avc-mpl GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "demo/app.py"]
