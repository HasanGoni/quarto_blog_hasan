FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy pillow
RUN useradd --no-create-home --uid 10001 --shell /usr/sbin/nologin runner
COPY runner.py /opt/runner.py
USER 10001
WORKDIR /scratch
ENTRYPOINT ["python", "-u", "/opt/runner.py"]
