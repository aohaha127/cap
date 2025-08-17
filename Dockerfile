# 第一阶段：构建环境
FROM python:3.9 as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 第二阶段：运行环境
FROM python:3.9-slim

# 设置时区和依赖
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglvnd0 \
    libgomp1 \
    ca-certificates \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 从构建阶段复制已安装的包
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH

# 暴露端口
EXPOSE 7777

# 运行项目
CMD ["python", "server.py"]
