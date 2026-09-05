#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorex 录像智能检测：识别 人 / 车 / 宠物，命中即保存带框快照并记录事件日志。

核心设计（改之前请先读懂）：
  1. **不占用摄像头 RTSP**：Lorex 只允许 1 路连接，已被 WD 录像占用。
     本脚本只分析「已录下来的文件」，绝不再拉流，不会影响正在进行的录制。
  2. **准实时**：录像已改为分片 MP4(fragmented)，未封口的在用文件也能被读取，
     因此可增量追帧，延迟 = 轮询间隔（默认 3 分钟）。
  3. **取文件方式**：UGREEN→WD 走免密 SSH（sshfs 在该 NAS 上无权限，已放弃挂载），
     实测 236MB 拉取约 3~5 秒。
  4. **只读不删**：对录像只做读取；快照与事件日志通过 smbclient 回写（匿名可写已验证）。
  5. **命中才落盘 + 同类冷却**：避免一个人走过存几十张。

用法：
  python3 lorex_detect.py --test [分钟数]     # 拉最新录像测一小段（不写状态、不存图）
  python3 lorex_detect.py --daemon            # 常驻循环检测
"""
import os
import re
import sys
import json
import time
import signal
import subprocess
from datetime import datetime, timedelta

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 OpenCV")

# ============================== 配置 ==============================
LOCAL_HOME = "/home/sunny/camara-detect"
MODEL_PATH = os.path.join(LOCAL_HOME, "yolov8n.onnx")
RACCOON_MODEL = os.path.join(LOCAL_HOME, "raccoon.onnx")   # fine-tune 加的浣熊单类模型
RACCOON_CONF_TH = 0.40
RACCOON_CLASS_ID = 80                                 # 与 COCO 0/15/16/17 不冲突
SSH_WD = os.path.join(LOCAL_HOME, "ssh_wd.sh")     # 兼容 WD 老算法的 ssh 包装
WD_HOST = "root@192.168.2.90"
REMOTE_REC = "/shares/Public/Lorex_Records"
SNAP_REMOTE_ROOT = "Lorex_Snapshots"               # 相对 WD 的 Public 共享
SMB_URL = "//192.168.2.90/Public"

WORK_DIR = "/tmp/camara_detect"
STATE_FILE = os.path.join(LOCAL_HOME, "state.json")
EVENT_DIR = os.path.join(LOCAL_HOME, "events")

SAMPLE_FPS = 1.0        # 每秒抽 1 帧做检测
CONF_TH = 0.45
NMS_TH = 0.50
COOLDOWN_SEC = 45       # 同一类别在此秒数内最多存 1 张
POLL_SEC = 180          # 守护模式轮询间隔
MAX_CHUNK_SEC = 1200    # 单次最多追多少秒

# 静态热区抑制：固定摄像头下，某些固定物体（石柱、灌木丛、信箱杆等）会被
# YOLOv8n 反复误标为 person。坐标是「原始帧坐标」(2560x1440)；只对 person 生效，
# 避免误伤 cat/dog 走过同一区域。热区是窄矩形，真人 bbox 远大于此、中心一般落不进。
# 2026-09-03 修正：原 y1=340 差 9px 罩不住石柱。COCO 实际给出的 person 框为
# (1386,240,1456,422)，中心 (1421,331) —— x 落在区间内但 y=331 < 340，中心判定漏出，
# 导致石柱被持续误报 person（0.25-0.62），污染统计并可能阻塞运动兜底。y1 上移至 220
# 以完整覆盖框顶（240）并留出余量。
EXCLUDE_ZONES = [
    (1270, 220, 1450, 770),   # 石柱：车道边石墩，从木槿花丛里露出来的一截
]

# ============================== 运动兜底 ==============================
# 用途：YOLOv8n (COCO 80 类) 没有 raccoon / opossum / skunk / fox / owl 等本地
# 野生动物。夜间用车前/垃圾桶区 ROI 内的帧差检测运动，配合 person/cat/dog 互斥
# （无 COCO 命中才触发），存 "animal" 快照（2026-09-03 起由 unknown_animal 简化而来）。
# 覆盖 COCO 漏掉的所有动物；浣熊模型暂disabled，未知动物统一走此兜底。
# 限制：只夜间（动物活动高峰）启用 + 60s 冷却 + 每 chunk 重置 prev_gray，
#       避免白天车辆/行人/树影误报和日→夜光照切换的误触发。
MOTION_ROI = (0, 200, 1280, 720)     # x1,y1,x2,y2，1280 宽缩放坐标
# 2026-09-05 抬升：之前 y1=380 把猫/浣熊活动带（y≈260-310）整个切掉；
# 全画面探测验证画面动态污染源（树梢/灌木）只到 y≈170，故 y1=200 安全。
MOTION_MIN_AREA = 500                 # ROI 内运动像素 > 此值才触发
MOTION_MIN_BBOX = 30                  # 最大轮廓 bbox 至少 30x30，过滤单点噪点
MOTION_COOLDOWN_SEC = 60              # animal 运动兜底保存冷却
MOTION_NIGHT_HOURS = set(range(20, 24)) | set(range(0, 6))  # 20:00-05:59
MOTION_BLUR_KSIZE = 5                 # 高斯模糊降噪（IR 夜视噪点）
MOTION_THRESH = 20                    # 帧差像素阈值(0-255)
MOTION_COLOR = (60, 200, 255)         # 兜底框颜色（橙黄，与 person/cat/dog 区分）
# 自适应背景 + 时间去抖参数（修复静止目标/壁灯干扰误报）
MOTION_BG_ALPHA = 0.02                # 背景 EMA 学习率（越小越慢，静止物体越快并入背景）
MOTION_DEBOUNCE_WINDOW = 4            # 去抖滑动窗口（帧）
MOTION_DEBOUNCE_HITS = 2              # 窗口内至少命中几次才算真运动（过滤单帧灯光闪烁）
# 照明阶跃抑制（2026-09-05 新增）：车库屋檐感应灯 + Lorex 自带 IR 灯在自动
# 感应触发时会照亮门前区域（拖车左侧+车库前）。这种「整片区域同步变亮/暗」
# 会被误判为大范围运动，bbox 圈出巨框（251770 px @ 23:19:26）且不触发
# MOTION_GLOBAL_FRAC（占比仅 0.477 < 0.6）。两个独立判据互补：
#   - 中位数阶跃：照明事件 ROI 灰度中位数前后帧跳变 ≥8
#   - 同向性：变化像素里 >80% 同号（真实运动约 0.4-0.55）
# 任一满足即判照明事件，强制把背景快速吸收到当前帧（fast alpha），跳过该帧
# 不进运动判定，避免污染去抖窗口和后续帧的「背景错位」漏检。
MOTION_LIGHT_STEP = 8.0               # 帧间 ROI 中位数变化阈值
MOTION_LIGHT_POS_FRAC = 0.80          # 正差异占比阈值（>此为亮起，<1-此为熄灭）
MOTION_LIGHT_AREA_MIN = 0.05          # 启用同向性判据的最小变化面积占比（太小可能是噪点）
MOTION_BG_FAST_ALPHA = 0.35            # 照明事件时背景快速吸收的学习率
MOTION_LIGHT_SKIP_MIN = 12             # LIGHT 触发后强制 skip 的最少帧数（生产 1fps ≈ 12s）
MOTION_LIGHT_SKIP_RENEW = 6            # skip 期间若 area 仍大则续命的最少帧数
MOTION_MIN_DENSITY = 0.08              # bbox 填充密度阈值：稀疏 bbox 是伪运动
                                       # （真动物 density > 0.3，光照颗粒/IR 噪声 < 0.05）
# 运动兜底屏蔽区（仅对 motion 兜底 animal 生效；COCO 对真实动物不受影响）
# 反推自 WD 夜间全画面 Garage_20260902_020540.mp4（@10s, 2026-09-02 02:05:50）：
#   L1 近壁灯: 灯泡 box ~x[180,420] y[0,230]，覆盖灯泡+顶部光晕
#   L2 远壁灯: 灯泡 box ~x[420,680] y[0,230]，覆盖灯泡+顶部光晕
#   蓝篷布:   HSV 蓝识别 x[1229,1959] y[441,856]；区间下沿取 890 防飘移，顶边 430
#             留 10px 余量盖住篷布上沿 drape。稳态的整面亮墙由自适应背景 EMA 吸收，
#             右侧若干静态亮斑（IR/反射）同理，无需整墙 zone。
MOTION_EXCLUDE_ZONES = [
    (180, 0, 420, 230),      # L1 近壁灯（左墙）灯泡+顶部光晕
    (420, 0, 680, 230),      # L2 远壁灯（左墙远端）灯泡+顶部光晕
    (1210, 430, 1970, 890),  # 蓝色篷布（车道上静态目标，双保险）
]
MOTION_GLOBAL_FRAC = 0.6              # ROI 内变化像素占比超此值视为全局亮度变化（壁灯/IR 漂移），非运动
# 运动兜底与 COCO 的互斥判定。原逻辑是 `not dets`（COCO 有任何命中就不兜底），语义本意是
# "模型已经认出来了就别重复兜底"，但静态误报（如石柱被判 person）同样会命中，于是把真实
# 动物的运动兜底一起堵死。改为几何判定：只有某个检测框确实落在运动区域内（被覆盖率达标）
# 才认为该运动已被模型覆盖。检测框与运动框不相干时（位置分离）照常兜底。
MOTION_COVER_FRAC = 0.30              # 检测框被运动区域覆盖的面积占比 >= 此值，视为已覆盖


class MotionDetector:
    """自适应背景建模 + 时间去抖 + 照明阶跃抑制的帧差兜底。
    相比朴素 absdiff(prev,cur)：
      - 维护缓慢更新的背景 EMA，静止物体（车道蓝色篷布拖车、信箱等）很快并入背景，
        不再产生帧差；夜间大门壁灯造成的缓慢亮度漂移也被背景吸收。
      - 仅在「背景之外出现持续运动」时判定 motion，配合时间去抖（debounce），
        单帧灯光闪烁 / IR illumination 抖动不会触发快照。
      - 照明阶跃抑制：车库屋檐感应灯 + Lorex IR 补光灯自动感应时，会把门前区域
        整片照亮（同步变亮或同步变暗）。这类「同向变化」的差分会被当成巨框运动，
        用两个独立判据捕捉后强制把背景快速吸收，避免污染后续帧。
    背景更新策略（2026-09-05 调整）：原逻辑是「运动时不更新背景」——灯亮期间
    持续大面积变化 → 背景卡死在「灯灭」状态 → 灯亮的每一帧都判运动（已废）。
    现在改为「**总是用正常 alpha 更新**」，理由：慢 alpha（0.005）在持续小幅差分
    下永远追不上，会卡死在伪运动（i=64 实测 bbox density 仅 0.02）。正常 alpha
    能消化光照余热（EMA 系数 0.02 → 50 帧 ≈ 50 秒 1fps 追平），动物穿过几帧
    就离开 ROI，期间 area 会快速衰减不会被吸收。
    每 chunk 由调用方 new 一个实例，避免日/夜切换污染背景。"""
    def __init__(self, alpha=MOTION_BG_ALPHA, debounce_hits=MOTION_DEBOUNCE_HITS,
                 debounce_window=MOTION_DEBOUNCE_WINDOW):
        self.bg = None                       # float32 背景（灰度，1280 宽空间）
        self.alpha = alpha
        self.debounce_hits = debounce_hits
        self.debounce_window = debounce_window
        self._hits = []                      # 滑动窗口：最近若干帧是否触发
        self._prev_med = None                # 上帧 ROI 灰度中位数（照明阶跃判据）
        self._light_skip = 0                 # 照明事件后续强制快速吸收的剩余帧数

    def _record(self, triggered):
        self._hits.append(triggered)
        if len(self._hits) > self.debounce_window:
            self._hits.pop(0)
        return triggered

    def update(self, frame):
        """返回 (motion_detected, bbox, area)。bbox 在 1280 宽缩放空间。"""
        h, w = frame.shape[:2]
        s = 1280.0 / w
        small = cv2.resize(frame, (1280, int(h * s)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)
        rx1, ry1, rx2, ry2 = MOTION_ROI
        ry2 = min(ry2, small.shape[0])
        gray_f = gray.astype(np.float32)
        med_roi = float(np.median(gray[ry1:ry2, rx1:rx2]))
        if self.bg is None:
            self.bg = gray_f.copy()
            self._prev_med = med_roi
            return False, None, 0
        diff = np.abs(self.bg - gray_f).astype(np.uint8)
        _, thresh = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        roi = thresh[ry1:ry2, rx1:rx2]
        area = int((roi > 0).sum())
        roi_total = max(1, (rx2 - rx1) * (ry2 - ry1))
        frac = area / roi_total
        # 全局亮度变化判据（兜底，防止单判据漏检）
        is_global = frac > MOTION_GLOBAL_FRAC

        # 照明阶跃判据 1：ROI 灰度中位数前后帧跳变
        dmed = med_roi - (self._prev_med if self._prev_med is not None else med_roi)
        light_step_hit = abs(dmed) >= MOTION_LIGHT_STEP

        # 照明阶跃判据 2：变化像素同向性（正差异占比极高=全亮起；极低=全暗）
        signed_roi = (gray_f - self.bg)[ry1:ry2, rx1:rx2]
        pos_sum = float(np.clip(signed_roi, 0, None).sum())
        neg_sum = float(np.clip(-signed_roi, 0, None).sum())
        pos_frac = pos_sum / (pos_sum + neg_sum + 1e-6)
        light_directional_hit = (frac >= MOTION_LIGHT_AREA_MIN and
                                  (pos_frac >= MOTION_LIGHT_POS_FRAC or
                                   pos_frac <= (1.0 - MOTION_LIGHT_POS_FRAC)))

        is_light = light_step_hit or light_directional_hit

        # 照明事件或其后余热：强制用快 alpha 把背景拉向当前帧，避免污染后续帧；
        # 标记为「无运动」并清空去抖窗口（防「1帧灯+1帧真运动」拼成 2/4 假阳性）。
        # _light_skip 起跳设到 MOTION_LIGHT_SKIP_MIN（生产 1fps ≈ 12s），确保 LIGHT
        # 后期未触发的「半更新态」（背景错位→颗粒噪声）也在保护期内被快吸收。
        if is_light:
            self.bg = ((1.0 - MOTION_BG_FAST_ALPHA) * self.bg
                       + MOTION_BG_FAST_ALPHA * gray_f)
            self._light_skip = max(self._light_skip, MOTION_LIGHT_SKIP_MIN)
            self._hits.clear()
            self._prev_med = med_roi
            return False, None, area

        if self._light_skip > 0:
            self.bg = ((1.0 - MOTION_BG_FAST_ALPHA) * self.bg
                       + MOTION_BG_FAST_ALPHA * gray_f)
            # skip 期间若差分仍大（半更新态/光照余热），续命防止提前退出误报
            if area >= MOTION_MIN_AREA:
                self._light_skip = max(self._light_skip, MOTION_LIGHT_SKIP_RENEW)
            self._light_skip -= 1
            self._prev_med = med_roi
            return False, None, area

        # 背景更新：总是用正常 alpha（见类 docstring 解释慢 alpha 已废）
        self.bg = (1.0 - self.alpha) * self.bg + self.alpha * gray_f
        self._prev_med = med_roi

        if is_global:
            return self._record(False), None, area
        if area < MOTION_MIN_AREA:
            return self._record(False), None, area
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._record(False), None, area
        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        if bw < MOTION_MIN_BBOX or bh < MOTION_MIN_BBOX:
            return self._record(False), None, area
        # bbox 密度过滤：真动物 bbox 致密（density > 0.3），照明颗粒/IR 噪声稀疏
        # （density < 0.05，i=64 实测 0.02）。稀疏 bbox 直接丢弃，不进去抖窗口。
        bb_area = max(1, bw * bh)
        if (area / bb_area) < MOTION_MIN_DENSITY:
            return self._record(False), None, area
        # 时间去抖：本帧有运动，需滑动窗口内累计命中达标才判定为真运动
        if sum(self._hits) + 1 < self.debounce_hits:
            self._record(True)
            return False, None, area
        self._record(True)
        return True, (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh), area


# 最终输出类目：person / cat / dog + 浣熊（fine-tune 模型，COCO 无此类）。
# 注：raccoon 于 2026-09-03 暂时禁用（见 main()），未知动物统一走 motion 兜底 "animal"。
WANTED = {
    0: "person", 16: "cat", 17: "dog", RACCOON_CLASS_ID: "raccoon",
}
# 检测候选类：在 WANTED 基础上保留 bird(15) 作为「内部候选」。
# 原因：YOLOv8n 偶尔把清晰猫误判为 bird(15)，correct_animal_class 须先见到该框才能救回成 cat/dog；
# 真正「决定性 bird」（纠错未触发）会在 detect() 末尾丢弃，因为用户只需 person/cat/dog。
CANDIDATE_CLASSES = {0, 15, 16, 17}
COLOR = {"person": (60, 220, 60), "cat": (200, 120, 255), "dog": (80, 160, 255),
         "raccoon": (0, 0, 200)}

_running = True


def _stop(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


# ============================== 状态 ==============================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_FILE)


# ============================== 远程访问 ==============================
def ssh_wd(cmd, timeout=120):
    r = subprocess.run([SSH_WD, WD_HOST, cmd], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("ssh 失败: %s" % (r.stderr.strip()[:120] or cmd[:80]))
    return r.stdout


def list_remote_segments():
    out = ssh_wd("find %s -name 'Garage_*.mp4' | sort" % REMOTE_REC, timeout=180)
    return [l.strip() for l in out.splitlines() if l.strip().endswith(".mp4")]


def fetch(remote, local):
    """优先走 smbclient（实测 236MB 约 3 秒，比 ssh cat 快约 9 倍），失败回退 ssh cat"""
    if os.path.exists(local):
        os.remove(local)
    rel = remote.split("/shares/Public/", 1)[-1].replace("/", "\\")
    try:
        r = subprocess.run(["smbclient", SMB_URL, "-N", "-c",
                            "get \"%s\" \"%s\"" % (rel, local)],
                           capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and os.path.exists(local) and os.path.getsize(local) > 1024:
            return True
    except Exception:
        pass
    with open(local, "wb") as f:
        subprocess.run([SSH_WD, WD_HOST, "cat '%s'" % remote], stdout=f,
                       stderr=subprocess.DEVNULL, timeout=900)
    return os.path.exists(local) and os.path.getsize(local) > 1024


# ============================== 分类纠错 ==============================
# 已知问题：YOLOv8n 在「清晰猫」上偶尔把 bird(15) 误判为 top 类。
# 室内/猫家庭场景下 bird 几乎不可能出现，而 cat(16)/dog(17) 与 bird 形态易混。
# 若 top==bird 但 cat/dog 分数接近且可信，则改判为更接近的动物，专治 cat→bird 误标。
BIRD_CORRECT_MARGIN = 0.20   # bird 与第二候选动物分差小于此值才改判
BIRD_CORRECT_FLOOR = 0.25    # 候选动物分数至少达到此值才可信


def correct_animal_class(cls_id, conf, score_row):
    if cls_id != 15:                      # 只处理 bird 误标，避免 cat/dog 互相翻错
        return cls_id, conf
    cat_s = float(score_row[16])
    dog_s = float(score_row[17])
    alt = 16 if cat_s >= dog_s else 17    # 取 cat/dog 中分数更高者
    alt_s = max(cat_s, dog_s)
    if alt_s >= BIRD_CORRECT_FLOOR and (conf - alt_s) < BIRD_CORRECT_MARGIN:
        return alt, alt_s
    return cls_id, conf


def in_excluded_zone(box, zones=None):
    """检测中心点是否落入任一热区。box=(x1,y1,x2,y2) 原始帧坐标。"""
    zs = zones if zones is not None else EXCLUDE_ZONES
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for zx1, zy1, zx2, zy2 in zs:
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            return True
    return False


def motion_covered_by_dets(mbox, dets, cover_frac=MOTION_COVER_FRAC):
    """运动 bbox 是否已被某个检测框覆盖（几何判定，替代原来的 `not dets`）。

    mbox 来自 MotionDetector，位于 1280 宽缩放空间；dets 的框位于原始帧空间
    （2560x1440），故先把 mbox 放大 2 倍对齐。判定用「检测框被运动区域覆盖的面积
    占比」而非 IoU：动物的检测框通常整体落在运动框内部，IoU 会被大运动框稀释，
    而覆盖率能稳定接近 1。
    """
    if mbox is None or not dets:
        return False
    mx1, my1, mx2, my2 = [int(v) * 2 for v in mbox]
    mw, mh = max(1, mx2 - mx1), max(1, my2 - my1)
    for cid, conf, (bx1, by1, bx2, by2) in dets:
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        ix1, iy1 = max(mx1, bx1), max(my1, by1)
        ix2, iy2 = min(mx2, bx2), min(my2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            continue                      # 完全不相干（如石柱误报 vs 车道上的动物）
        if inter / float(bw * bh) >= cover_frac:
            return True
    return False


# ============================== YOLOv8 ==============================
class Detector:
    def __init__(self, model_path, conf_th=CONF_TH, nms_th=NMS_TH, class_map=None):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.size = 640
        self.zone_skips = 0     # 本段录像被热区抑制掉的 person 数（守护模式在段末打印）
        # class_map: 模型类 id -> (我们的 cid, 名称)。用于单类 fine-tune 模型（如浣熊）。
        # 若设置，detect() 跳过 COCO 专属逻辑（bird 纠错 / 石柱热区 / 候选类过滤），直接按 class_map 重映射。
        self.class_map = class_map

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.size / w, self.size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        dw, dh = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, r, dw, dh

    def detect(self, frame):
        canvas, r, dw, dh = self.letterbox(frame)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        pred = self.net.forward()[0].transpose(1, 0)     # (8400, 84)

        boxes = pred[:, :4]
        # 关键：不同导出的 ONNX 输出可能是「原始 logits」也可能「已 sigmoid」。
        # 已激活的值全部落在 [0,1]；若再套一次 sigmoid 会把所有置信度压向 0.5，
        # 导致每一帧都虚报所有类别 —— 必须按取值范围自动判断。
        raw = pred[:, 4:]
        scores = self._sigmoid(raw) if (raw.min() < 0 or raw.max() > 1) else raw
        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)

        if self.class_map is not None:
            # 单类 fine-tune 模型：仅按置信度过滤（所有类都想要）
            keep = (confs > self.conf_th)
        else:
            keep = (confs > self.conf_th) & np.isin(cls_ids, list(CANDIDATE_CLASSES))
        if not np.any(keep):
            return []
        boxes = boxes[keep]
        cls_ids = cls_ids[keep]
        confs = confs[keep]
        scores = scores[keep]          # 保留整行分类分数，供分类纠错使用

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - bw / 2 - dw) / r
        y1 = (cy - bh / 2 - dh) / r
        x2 = (cx + bw / 2 - dw) / r
        y2 = (cy + bh / 2 - dh) / r
        H, W = frame.shape[:2]
        np.clip(x1, 0, W - 1, out=x1); np.clip(x2, 0, W - 1, out=x2)
        np.clip(y1, 0, H - 1, out=y1); np.clip(y2, 0, H - 1, out=y2)

        rects = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
                 for i in range(len(boxes))]
        idx = cv2.dnn.NMSBoxes(rects, [float(c) for c in confs], self.conf_th, self.nms_th)
        if idx is None or len(idx) == 0:
            return []
        out = []
        for i in np.array(idx).flatten():
            cid, conf = int(cls_ids[i]), float(confs[i])
            if self.class_map is not None:
                # 单类 fine-tune 模型：直接按 class_map 重映射，不做 COCO 专属处理
                if cid in self.class_map:
                    rcid, _ = self.class_map[cid]
                    rx, ry, rw, rh = rects[i]
                    out.append((rcid, conf,
                                (int(rx), int(ry), int(rx + rw), int(ry + rh))))
                continue
            cid, conf = correct_animal_class(cid, conf, scores[i])   # 分类纠错
            if cid == 15:            # 未被纠错救回的决定性 bird：用户只需 person/cat/dog，丢弃
                continue
            rx, ry, rw, rh = rects[i]
            bx1, by1, bx2, by2 = int(rx), int(ry), int(rx + rw), int(ry + rh)
            if cid == 0 and in_excluded_zone((bx1, by1, bx2, by2)):
                self.zone_skips += 1
                continue             # 静态热区抑制：固定摄像头下固定物体反复误标（如石柱）
            out.append((cid, conf, (bx1, by1, bx2, by2)))
        return out


# ============================== 抽帧 ==============================
def probe_size(path):
    """逐行解析 width=/height=，避免 codec_name 混入导致 split('x') 出错"""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height",
                            "-of", "default=noprint_wrappers=1", path],
                           capture_output=True, text=True, timeout=60)
        w = h = None
        for line in r.stdout.splitlines():
            if line.startswith("width="):
                w = int(line.split("=", 1)[1])
            elif line.startswith("height="):
                h = int(line.split("=", 1)[1])
        if w and h:
            return w, h
    except Exception:
        pass
    return 2560, 1440


def frames(path, start_sec, fps, width, height, end_sec=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % start_sec, "-i", path, "-an",
           "-vf", "fps=%.4f" % fps, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    if end_sec:
        cmd[cmd.index("-i") + 1:cmd.index("-i") + 1] = []
        cmd = ["ffmpeg", "-v", "error", "-ss", "%.3f" % start_sec, "-i", path,
               "-t", "%.3f" % (end_sec - start_sec), "-an",
               "-vf", "fps=%.4f" % fps, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10 ** 8)
    n = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if not buf or len(buf) < n:
                break
            yield i, np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=30)


# ============================== 输出 ==============================
def ensure_remote_dir(rel_dir):
    cur = ""
    for p in [x for x in rel_dir.split("/") if x]:
        cur = (cur + "/" + p) if cur else p
        subprocess.run(["smbclient", SMB_URL, "-N", "-c", "mkdir \"%s\"" % cur],
                       capture_output=True, text=True, timeout=30)


def smb_put(local_path, rel_remote_path):
    rel_dir = os.path.dirname(rel_remote_path)
    if rel_dir:
        ensure_remote_dir(rel_dir)
    r = subprocess.run(["smbclient", SMB_URL, "-N", "-c",
                        "cd \"%s\"; put \"%s\" \"%s\"" %
                        (rel_dir, local_path, os.path.basename(rel_remote_path))],
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0


def annotate(frame, dets):
    img = frame
    if img.shape[1] > 1280:
        s = 1280 / img.shape[1]
        img = cv2.resize(img, (1280, int(img.shape[0] * s)))
        dets = [(c, cf, (int(b[0] * s), int(b[1] * s), int(b[2] * s), int(b[3] * s)))
                for c, cf, b in dets]
    for cid, conf, (x1, y1, x2, y2) in dets:
        name = WANTED.get(cid, str(cid))
        color = COLOR.get(name, (60, 200, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = "%s %.2f" % (name, conf)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def append_event(day_str, row):
    os.makedirs(EVENT_DIR, exist_ok=True)
    path = os.path.join(EVENT_DIR, "%s.csv" % day_str)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("timestamp,class,confidence,snapshot,segment,offset_sec\n")
        f.write(",".join(str(x) for x in row) + "\n")
    return path


def upload_events(day_str):
    local = os.path.join(EVENT_DIR, "%s.csv" % day_str)
    if os.path.exists(local):
        smb_put(local, "%s/events/%s.csv" % (SNAP_REMOTE_ROOT, day_str))


# ============================== 主处理 ==============================
def seg_start_time(path):
    """Garage_YYYYMMDD_HHMMSS.mp4 -> datetime。
    注意：日期与时间被下划线分开，必须把两段数字拼起来再解析，
    只取 split('_')[0] 会只剩 8 位日期而解析失败（进而错用当前时间）。"""
    base = os.path.basename(path)
    m = re.search(r"Garage_(\d{8})_(\d{6})", base)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None
    return None


def process_file(det, local_path, remote_name, start_offset, st_time,
                 state=None, dry_run=False, max_sec=None, raccoon_det=None):
    W, H = probe_size(local_path)
    limit = max_sec or MAX_CHUNK_SEC
    remaining = limit
    cur_off = start_offset
    last_saved = {}
    last_motion_save = -1e9
    n_frames = n_hits = motion_hits = raccoon_hits = 0
    os.makedirs(WORK_DIR, exist_ok=True)
    day_str = (st_time or datetime.now()).strftime("%Y-%m-%d")

    while _running and remaining > 0:
        chunk = min(300.0, remaining)
        got = 0
        motion = MotionDetector()    # 每 chunk 重置，避免日/夜切换污染 prev_gray
        for idx, frame in frames(local_path, cur_off, SAMPLE_FPS, W, H,
                                 end_sec=cur_off + chunk):
            if not _running:
                break
            t_sec = cur_off + idx / SAMPLE_FPS
            got += 1
            ev_time = (st_time + timedelta(seconds=t_sec)) if st_time else datetime.now()
            day_str = ev_time.strftime("%Y-%m-%d")
            is_night = ev_time.hour in MOTION_NIGHT_HOURS

            dets = det.detect(frame)
            if raccoon_det is not None:
                rd = raccoon_det.detect(frame)
                if rd:
                    raccoon_hits += len(rd)
                    dets = dets + rd
            if dets:
                for cid, conf, box in dets:
                    cls_name = WANTED.get(cid, str(cid))
                    if t_sec - last_saved.get(cls_name, -1e9) < COOLDOWN_SEC:
                        continue
                    last_saved[cls_name] = t_sec
                    n_hits += 1
                    if dry_run:
                        log("  [TEST] %s %.2f @ %.1fs" % (cls_name, conf, t_sec))
                        continue
                    stamp = ev_time.strftime("%Y%m%d_%H%M%S")
                    img = annotate(frame, [(cid, conf, box)])
                    lj = os.path.join(WORK_DIR, "%s_%s_%.2f.jpg" % (stamp, cls_name, conf))
                    cv2.imwrite(lj, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    rel = "%s/%s/%s_%s_%.2f.jpg" % (SNAP_REMOTE_ROOT,
                                                    ev_time.strftime("%Y/%m/%d"),
                                                    stamp, cls_name, conf)
                    if smb_put(lj, rel):
                        append_event(day_str, (ev_time.strftime("%Y-%m-%d %H:%M:%S"),
                                               cls_name, "%.2f" % conf,
                                               os.path.basename(rel), remote_name,
                                               "%.1f" % t_sec))
                        log("  快照已存: %s (%s %.2f)" % (os.path.basename(rel), cls_name, conf))
                    else:
                        log("  快照上传失败: %s" % rel)
                    try:
                        os.remove(lj)
                    except Exception:
                        pass

            # 运动兜底：夜间 ROI 内有运动且无 WANTED 命中 → animal
            if is_night:
                mot, mbox, marea = motion.update(frame)
                if mot:
                    mx1, my1, mx2, my2 = mbox
                    # 运动兜底额外屏蔽区（原始 2560x1440 坐标）：壁灯/篷布等固定光源或物体
                    if in_excluded_zone((mx1 * 2, my1 * 2, mx2 * 2, my2 * 2),
                                        MOTION_EXCLUDE_ZONES):
                        mot = False
                # 互斥判定：原为 `not dets`（COCO 有任何命中就不兜底），静态误报
                # （石柱→person）会连带堵死真实动物的兜底。改为几何覆盖判定。
                covered = motion_covered_by_dets(mbox, dets) if mot else False
                if mot and not covered and (t_sec - last_motion_save) >= MOTION_COOLDOWN_SEC:
                    last_motion_save = t_sec
                    motion_hits += 1
                    if dry_run:
                        log("  [TEST] animal motion=%dpx @ %.1fs" % (marea, t_sec))
                    else:
                        stamp = ev_time.strftime("%Y%m%d_%H%M%S")
                        img = annotate(frame, [])
                        bx1, by1, bx2, by2 = mbox
                        (tw, th), _ = cv2.getTextSize("animal",
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), MOTION_COLOR, 3)
                        label = "animal motion=%dpx" % marea
                        cv2.rectangle(img, (bx1, max(0, by1 - th - 10)),
                                      (bx1 + tw + 8, by1), MOTION_COLOR, -1)
                        cv2.putText(img, label, (bx1 + 4, by1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20),
                                    2, cv2.LINE_AA)
                        lj = os.path.join(WORK_DIR, "%s_animal_%d.jpg" %
                                          (stamp, marea))
                        cv2.imwrite(lj, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        rel = "%s/%s/%s_animal_%d.jpg" % (
                            SNAP_REMOTE_ROOT, ev_time.strftime("%Y/%m/%d"),
                            stamp, marea)
                        if smb_put(lj, rel):
                            append_event(day_str,
                                         (ev_time.strftime("%Y-%m-%d %H:%M:%S"),
                                          "animal", "%d" % marea,
                                          os.path.basename(rel), remote_name,
                                          "%.1f" % t_sec))
                            log("  兜底快照: %s (运动 %dpx @ %s)" %
                                (os.path.basename(rel), marea,
                                 ev_time.strftime("%H:%M:%S")))
                        else:
                            log("  兜底快照上传失败: %s" % rel)
                        try:
                            os.remove(lj)
                        except Exception:
                            pass
        n_frames += got
        if got == 0:
            break
        cur_off += chunk
        remaining -= chunk
        if state is not None:
            state[remote_name] = cur_off
            save_state(state)

    if state is not None:
        state[remote_name] = max(state.get(remote_name, 0.0), cur_off)
        save_state(state)
        upload_events(day_str)
    zone_skips = getattr(det, "zone_skips", 0)
    det.zone_skips = 0
    log("  %s: %d 帧, 命中 %d 次(含浣熊 %d), 热区丢弃 %d 次, 运动兜底 %d 次, 推进到 %.1fs" %
        (remote_name, n_frames, n_hits, raccoon_hits, zone_skips, motion_hits, cur_off))
    return cur_off


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--daemon"

    if not os.path.exists(MODEL_PATH):
        sys.exit("模型不存在: %s" % MODEL_PATH)
    if not os.path.exists(SSH_WD):
        sys.exit("SSH 包装脚本不存在: %s（先跑 setup_ugreen.sh）" % SSH_WD)

    det = Detector(MODEL_PATH)
    log("模型已加载")
    # 浣熊模型：暂时禁用（2026-09-03，用户决策）。
    #   背景：v2_ir 两夜实测真实捕获 0、误报 4（汽车大灯/静止物体/人头部被误判为 raccoon）。
    #   决策：先统一用 motion 兜底 labeled "animal" 覆盖未知动物；待补负样本重训后再恢复。
    #   恢复：将下方 RACCOON_ENABLE 改为 True，并确保 %s 存在即可自动加载。
    RACCOON_ENABLE = False
    raccoon_det = None
    if RACCOON_ENABLE and os.path.exists(RACCOON_MODEL):
        try:
            raccoon_det = Detector(
                RACCOON_MODEL, conf_th=RACCOON_CONF_TH,
                class_map={0: (RACCOON_CLASS_ID, "raccoon")})
            log("浣熊模型已加载: %s (conf=%.2f)" % (RACCOON_MODEL, RACCOON_CONF_TH))
        except Exception as e:
            log("浣熊模型加载失败，仅用 COCO: %s" % e)
            raccoon_det = None
    else:
        log("浣熊模型已禁用（暂时）；未知动物改由 motion 兜底 'animal' 覆盖")

    if mode in ("--test", "--run"):
        dry = (mode == "--test")          # --test 只报不存；--run 会真正存快照
        minutes = 5.0
        remote = None
        for a in sys.argv[2:]:
            if "/" in a:
                remote = a                # 指定远程文件
            else:
                try:
                    minutes = float(a)
                except ValueError:
                    pass
        if remote is None:
            segs = list_remote_segments()
            if not segs:
                sys.exit("WD 上找不到录像文件")
            remote = segs[-1]
        log("%s %s (前 %.0f 分钟)" %
            ("测试(不存图)" if dry else "实跑(会存图)", os.path.basename(remote), minutes))
        local = os.path.join(WORK_DIR, "test.mp4")
        os.makedirs(WORK_DIR, exist_ok=True)
        t = time.time()
        if not fetch(remote, local):
            sys.exit("拉取文件失败")
        log("  拉取完成 %.1fMB / %.1fs" % (os.path.getsize(local) / 1e6, time.time() - t))
        process_file(det, local, os.path.basename(remote), 0.0,
                     seg_start_time(remote),
                     state=None if dry else load_state(),
                     dry_run=dry, max_sec=minutes * 60,
                     raccoon_det=raccoon_det)
        return

    if mode != "--daemon":
        print(__doc__)
        return

    log("守护模式启动，轮询 %ds" % POLL_SEC)
    state = load_state()
    while _running:
        try:
            segs = list_remote_segments()
            if not segs:
                log("未找到录像")
            else:
                remote = segs[-1]
                name = os.path.basename(remote)
                if state.get("__last") != name:
                    log("新片段: %s" % name)
                    state["__last"] = name
                    state.pop(name, None)
                    save_state(state)
                off = state.get(name, 0.0)
                local = os.path.join(WORK_DIR, "cur.mp4")
                os.makedirs(WORK_DIR, exist_ok=True)
                t = time.time()
                if fetch(remote, local):
                    log("拉取 %s (%.1fMB/%.1fs) 从 %.1fs 开始处理"
                        % (name, os.path.getsize(local) / 1e6, time.time() - t, off))
                    process_file(det, local, name, off,
                                 seg_start_time(remote), state,
                                 raccoon_det=raccoon_det)
                else:
                    log("拉取失败: %s" % name)
        except Exception as e:
            log("循环异常: %s" % e)
        for _ in range(POLL_SEC):
            if not _running:
                break
            time.sleep(1)
    log("已退出")


if __name__ == "__main__":
    main()
