FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        bash \
        openssh-server \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
        numpy scipy matplotlib

RUN mkdir /var/run/sshd && \
    echo 'root:cogs205bspring26' | chpasswd

WORKDIR /app
COPY bayes_factor.py .
COPY tests/ ./tests/

EXPOSE 22

CMD ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]