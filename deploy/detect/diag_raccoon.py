#!/usr/bin/env python3
# 诊断：浣熊模型在 IR 夜视录像上到底有没有信号（低阈值扫描）
import sys, os
sys.path.insert(0, '/home/sunny/camara-detect')
import numpy as np
import cv2
import lorex_detect as L

rdet = L.Detector(L.RACCOON_MODEL, conf_th=0.01, class_map={0: (L.RACCOON_CLASS_ID, "raccoon")})

local = "/tmp/raccoon_val.mp4"
W, H = L.probe_size(local)
print("WxH=%dx%d" % (W, H))

print("=== LOW-THRESHOLD scan on raccoon window (1600-1720s) ===")
max_conf_ever = 0.0
any_signal_frames = 0
for idx, frame in L.frames(local, 1600.0, L.SAMPLE_FPS, W, H, end_sec=1720.0):
    t = 1600.0 + idx / L.SAMPLE_FPS
    dets = rdet.detect(frame)
    if dets:
        any_signal_frames += 1
        for cid, conf, box in dets:
            if conf > max_conf_ever:
                max_conf_ever = conf
            print("  @%.1fs conf=%.3f box=%s" % (t, conf, box))

print("=== SUMMARY (thresh=0.01) ===")
print("frames with ANY raccoon signal:", any_signal_frames)
print("max conf seen in window:", round(max_conf_ever, 4))

# 同时用 COCO 模型看同窗口有无 animal 误判信号
det = L.Detector(L.MODEL_PATH)
print("=== COCO raw signal (thresh=0.1) on same window ===")
for idx, frame in L.frames(local, 1600.0, L.SAMPLE_FPS, W, H, end_sec=1720.0):
    t = 1600.0 + idx / L.SAMPLE_FPS
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640,640), swapRB=True, crop=False)
    det.net.setInput(blob)
    raw = det.net.forward()[0].transpose(1,0)
    scores = raw[:, 4:]
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    # 看 animal 类 (cat=16, dog=17, bird=15, bear=21) 的最高分
    animal_mask = np.isin(cls, [15,16,17,21])
    if animal_mask.any():
        am = np.where(animal_mask)[0]
        best = am[conf[am].argmax()]
        print("  @%.1fs COCO animal cls=%d conf=%.3f" % (t, cls[best], conf[best]))
