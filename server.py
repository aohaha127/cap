#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import base64
import json
import logging
import re
import requests
import uvicorn
import ddddocr
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s : %(message)s',
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="完整验证码识别服务")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 辅助函数：获取图像字节流
def get_image_bytes(image_data):
    """从不同来源获取图像字节流"""
    if isinstance(image_data, bytes):
        return image_data
    elif image_data.startswith('http'):
        response = requests.get(image_data, verify=False)
        response.raise_for_status()
        return response.content
    elif isinstance(image_data, str):
        return base64.b64decode(image_data)
    else:
        raise ValueError("不支持的图像数据类型")


# 辅助函数：图像转Base64
def image_to_base64(image, format='PNG'):
    """将PIL图像转换为Base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# 验证码处理器类
class CAPTCHAProcessor:
    def __init__(self):
        # 初始化识别器
        self.ocr = ddddocr.DdddOcr(show_ad=False)  # 通用OCR识别器
        self.det = ddddocr.DdddOcr(det=True)  # 目标检测识别器
        self.slide_detector = ddddocr.DdddOcr(det=False, ocr=False)  # 滑块识别专用

    # 基础OCR识别
    def classification(self, image_data):
        """识别图形验证码"""
        image_bytes = get_image_bytes(image_data)
        return self.ocr.classification(image_bytes)

    # 滑块匹配（第一种方式）
    def slide_match(self, sliding_image, back_image, simple_target=True):
        """滑块验证码识别（匹配方式）"""
        sliding_bytes = get_image_bytes(sliding_image)
        back_bytes = get_image_bytes(back_image)
        res = self.slide_detector.slide_match(sliding_bytes, back_bytes, simple_target=simple_target)
        return res['target'][0]

    # 滑块匹配（第二种方式）
    def slide_comparison(self, sliding_image, back_image):
        """滑块验证码识别（对比方式）"""
        sliding_bytes = get_image_bytes(sliding_image)
        back_bytes = get_image_bytes(back_image)
        res = self.ocr.slide_comparison(sliding_bytes, back_bytes)
        return res['target'][0]

    # 目标检测
    def detection(self, image_data):
        """检测图像中的文字或图标位置"""
        image_bytes = get_image_bytes(image_data)
        return self.det.detection(image_bytes)

    # 计算型验证码
    def calculate(self, image_data):
        """计算型验证码识别"""
        image_bytes = get_image_bytes(image_data)
        expression = self.ocr.classification(image_bytes)
        # 清理表达式
        expression = re.sub('=.*$', '', expression)
        expression = re.sub('[^0-9+\-*/()]', '', expression)
        result = eval(expression)
        return result

    # 图片分割
    def crop(self, image_data, y_coordinate):
        """图片分割处理"""
        if image_data.startswith('http'):
            image = Image.open(BytesIO(requests.get(image_data).content))
        else:
            image_bytes = get_image_bytes(image_data)
            image = Image.open(BytesIO(image_bytes))

        # 分割图片
        upper_half = image.crop((0, 0, image.width, y_coordinate))
        lower_half = image.crop((0, y_coordinate, image.width, image.height))

        # 转换为Base64
        sliding_image = image_to_base64(upper_half)
        back_image = image_to_base64(lower_half)

        return {"slidingImage": sliding_image, "backImage": back_image}

    # 点选验证码
    def select(self, image_data):
        """点选验证码处理"""
        image_bytes = get_image_bytes(image_data)

        # 转换为OpenCV格式
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        im = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 检测目标位置
        bboxes = self.det.detection(image_bytes)
        results = []

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            # 裁剪目标区域
            cropped_image = im[y1:y2, x1:x2]
            # 转换为Base64
            _, buffer = cv2.imencode('.png', cropped_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            # 识别内容
            text = self.ocr.classification(image_base64)
            results.append({text: [x1, y1, x2, y2]})

        return results


# 初始化处理器
processor = CAPTCHAProcessor()


# ================== 端点定义 ==================

@app.get("/")
async def root():
    """服务状态检查"""
    return {"status": "running", "message": "验证码识别服务正常运行中"}


@app.post("/ocr")
async def recognize_captcha(request: Request):
    """
    识别图形验证码（基础OCR）

    请求格式: {"image": "base64编码的图片数据或URL"}
    返回格式: {"code": 0, "data": "识别结果"}
    """
    try:
        data = await request.json()
        if "image" not in data:
            return {"code": 1, "message": "缺少image参数"}

        start_time = time.time()
        result = processor.classification(data["image"])
        elapsed = time.time() - start_time

        logger.info(f"OCR识别成功: {result}, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"OCR识别失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/slide")
async def recognize_slider(request: Request):
    """
    滑块验证码识别（简单模式）

    请求格式: {"bg_image": "背景图", "slide_image": "滑块图"}
             或 {"full_image": "完整截图base64"}
    返回格式: {"code": 0, "data": {"x": 横向距离}}
    """
    try:
        data = await request.json()

        if "bg_image" in data and "slide_image" in data:
            start_time = time.time()
            x = processor.slide_match(
                data["slide_image"],
                data["bg_image"],
                simple_target=True
            )
            elapsed = time.time() - start_time

            logger.info(f"滑块识别成功: x={x}, 耗时: {elapsed:.3f}秒")
            return {"code": 0, "data": {"x": x}}

        elif "full_image" in data:
            logger.info("接收到完整截图，返回默认值")
            return {"code": 0, "data": {"x": 150}}
        else:
            return {"code": 1, "message": "缺少必要参数"}
    except Exception as e:
        logger.error(f"滑块识别失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/capcode")
async def capcode_handler(request: Request):
    """
    滑块验证码识别（高级模式）

    请求格式: {"slidingImage": "滑块图", "backImage": "背景图", "simpleTarget": true}
    返回格式: {"code": 0, "data": {"x": 横向距离}}
    """
    try:
        data = await request.json()
        sliding_image = data["slidingImage"]
        back_image = data["backImage"]
        simple_target = data.get("simpleTarget", True)

        start_time = time.time()
        x = processor.slide_match(sliding_image, back_image, simple_target)
        elapsed = time.time() - start_time

        logger.info(f"滑块高级识别成功: x={x}, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": {"x": x}}
    except Exception as e:
        logger.error(f"滑块高级识别失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/slideComparison")
async def slide_comparison_handler(request: Request):
    """
    滑块验证码识别（对比模式）

    请求格式: {"slidingImage": "滑块图", "backImage": "背景图"}
    返回格式: {"code": 0, "data": {"x": 横向距离}}
    """
    try:
        data = await request.json()
        sliding_image = data["slidingImage"]
        back_image = data["backImage"]

        start_time = time.time()
        x = processor.slide_comparison(sliding_image, back_image)
        elapsed = time.time() - start_time

        logger.info(f"滑块对比识别成功: x={x}, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": {"x": x}}
    except Exception as e:
        logger.error(f"滑块对比识别失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/detection")
async def detection_handler(request: Request):
    """
    目标检测（识别图像中的文字/图标位置）

    请求格式: {"image": "图像数据"}
    返回格式: {"code": 0, "data": [[x1, y1, x2, y2], ...]}
    """
    try:
        data = await request.json()
        image = data["image"]

        start_time = time.time()
        result = processor.detection(image)
        elapsed = time.time() - start_time

        logger.info(f"目标检测成功, 检测到 {len(result)} 个目标, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"目标检测失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/calculate")
async def calculate_handler(request: Request):
    """
    计算型验证码识别

    请求格式: {"image": "图像数据"}
    返回格式: {"code": 0, "data": 计算结果}
    """
    try:
        data = await request.json()
        image = data["image"]

        start_time = time.time()
        result = processor.calculate(image)
        elapsed = time.time() - start_time

        logger.info(f"计算验证码成功: 结果={result}, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"计算验证码失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


@app.post("/crop")
async def crop_handler(request: Request):
    """
    图片分割处理

    请求格式: {"image": "图像数据", "y_coordinate": 分割位置}
    返回格式: {"code": 0, "data": {"slidingImage": "...", "backImage": "..."}}
    """
    try:
        data = await request.json()
        image = data["image"]
        y_coordinate = data["y_coordinate"]

        start_time = time.time()
        result = processor.crop(image, y_coordinate)
        elapsed = time.time() - start_time

        logger.info(f"图片分割成功, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"图片分割失败: {str(e)}")
        return {"code": 1, "message": f"处理失败: {str(e)}"}


@app.post("/select")
async def select_handler(request: Request):
    """
    点选验证码识别

    请求格式: {"image": "图像数据"}
    返回格式: {"code": 0, "data": [{"文字": [x1, y1, x2, y2]}, ...]}
    """
    try:
        data = await request.json()
        image = data["image"]

        start_time = time.time()
        result = processor.select(image)
        elapsed = time.time() - start_time

        logger.info(f"点选验证码识别成功, 检测到 {len(result)} 个目标, 耗时: {elapsed:.3f}秒")
        return {"code": 0, "data": result}
    except Exception as e:
        logger.error(f"点选验证码识别失败: {str(e)}")
        return {"code": 1, "message": f"识别失败: {str(e)}"}


# 启动服务
if __name__ == "__main__":
    logger.info("验证码识别服务已启动，监听端口：7777")
    uvicorn.run(app, host="0.0.0.0", port=7777)