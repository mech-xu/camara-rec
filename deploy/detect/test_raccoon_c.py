#!/usr/bin/env python3
# 验证：对浣熊录像段同时跑 COCO 模型 + 浣熊 fine-tune 模型，对比抓取效果
import sys, os
sys.path.insert(0, '/home/sunny/camara-detect')
import datetime
import cv2
import lorex_detect as L

det = L.Detector(L.MODEL_PATH)
rdet = L.Detector(L.RACCOON_MODEL, conf_th=L.RACCOON_CONF_TH,
                  class_map={0: (L.RACCOON_CLASS_ID, "raccoon")})

remote = "/shares/Public/Lorex_Records/2026/09/01/Garage_20260901_003539.mp4"
local = "/tmp/raccoon_val.mp4"
print("== fetch %s ==" % remote)
L.fetch(remote, local)
W, H = L.probe_size(local)
st = L.seg_start_time(remote)
print("seg start=%s WxH=%dx%d" % (st, W, H))

START, END = 1500.0, 2000.0   # 覆盖 01:00:39 - 01:08:59（含浣熊窗口 ~1631s）
coco_hits = raccoon_hits = motion_hits = 0
saved = []
motion = L.MotionDetector()
for idx, frame in L.frames(local, START, L.SAMPLE_FPS, W, H, end_sec=END):
    t = START + idx / L.SAMPLE_FPS
    ev = st + datetime.timedelta(seconds=t)
    is_night = ev.hour in L.MOTION_NIGHT_HOURS

    dets = det.detect(frame)
    rd = rdet.detect(frame)
    if rd:
        raccoon_hits += len(rd)
        for cid, conf, box in rd:
            bx1, by1, bx2, by2 = box
            print("RACCOON @%.1fs conf=%.2f box=(%d,%d,%d,%d) W=%d H=%d"
                  % (t, conf, bx1, by1, bx2, by2, bx2 - bx1, by2 - by1))
            img = L.annotate(frame, [(cid, conf, box)])
            fn = "/tmp/raccoon_val_%.1f.jpg" % t
            cv2.imwrite(fn, img)
            saved.append(fn)
    if dets:
        for cid, conf, box in dets:
            coco_hits += 1
            print("COCO %s @%.1fs conf=%.2f" % (L.WANTED.get(cid, str(cid)), t, conf))
    if is_night:
        mot, mbox, marea = motion.update(frame)
        if mot and not dets and not rd:
            motion_hits += 1

print("=== SUMMARY ===")
print("frames in window=%d" % int((END - START) * L.SAMPLE_FPS))
print("COCO hits=%d  raccoon hits=%d  motion_fallback=%d" % (coco_hits, raccoon_hits, motion_hits))
print("raccoon snapshots saved=%d" % len(saved))
for s in saved:
    print("  %s" % s)
