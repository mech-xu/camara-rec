#!/bin/bash
# Lorex W482CAD RTSP -> WDMyCloud 本地存储
# 视频原画复制(-c:v copy) + 丢音频(-an) + 零转码。最省算力。
# 存储布局：Lorex_Records/YYYY/MM/DD/Garage_YYYYMMDD_HHMMSS.mp4（与 Reolink 约定一致）
set -u
export TZ=America/Toronto
BASE='/shares/Public/Lorex_Records'
LOG="$BASE/record.log"
# 敏感凭证从同目录 secrets.env 读取（该文件已 gitignore，不入库）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/secrets.env" ]; then
  source "$SCRIPT_DIR/secrets.env"
fi
# 单实例锁：原子 mkdir 锁目录（避开 flock / pgrep 在 WD BusyBox 上的不稳定与竞态）
LOCKDIR=/tmp/lorex_record.lockdir
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$(date) already running (lockdir exists), exit" >> "$LOG"
  exit 1
fi
echo $$ > "$LOCKDIR/pid"

# 日期目录值守：ffmpeg 的 segment muxer 不会自动创建 %Y/%m/%d 目录，
# 必须提前建好今天和明天的目录，否则跨零点开新片段时会 ENOENT 失败。
ensure_dirs() {
  mkdir -p "$BASE/$(date +%Y/%m/%d)" 2>/dev/null
  mkdir -p "$BASE/$(date -d '+1 day' +%Y/%m/%d)" 2>/dev/null
}
ensure_dirs
( while true; do ensure_dirs; sleep 600; done ) &
KEEPER=$!

# 停滞看门狗（关键）：
# 摄像头静默掉线时 ffmpeg 会阻塞在 recv() —— 不退出、不报错、**不写日志**，
# 进程仍在，因此 cron 看门狗（只判断进程是否存在）无法发现问题，录像会长时间静默冻结。
# 本看门狗改为按"当前片段是否还在增长"判断：每 2 分钟取最新片段，间隔 15 秒比大小，
# 若无增长则判定掉线，杀掉 ffmpeg 让下面 while 循环 5 秒后重连（会新建一段）。
( while true; do
    sleep 120
    F=$(ls -t "$BASE"/$(date +%Y/%m/%d)/Garage_*.mp4 2>/dev/null | head -1)
    [ -n "$F" ] || F=$(find "$BASE" -name 'Garage_*.mp4' -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    [ -n "$F" ] && [ -f "$F" ] || continue
    AGE=$(( $(date +%s) - $(stat -c %Y "$F" 2>/dev/null || echo 0) ))
    [ "$AGE" -ge 30 ] || continue          # 刚创建的新片段，跳过本轮
    A=$(stat -c %s "$F" 2>/dev/null)
    sleep 15
    B=$(stat -c %s "$F" 2>/dev/null)
    if [ -n "$A" ] && [ "$A" = "$B" ]; then
      echo "$(date) [stall-watchdog] 片段 15s 无增长 ($(basename $F) size=$A) -> 判定摄像头掉线，重启 ffmpeg" >> "$LOG"
      pkill -9 ffmpeg 2>/dev/null
      sleep 60
    fi
  done ) &
WATCHDOG=$!

trap 'kill -9 $KEEPER $WATCHDOG 2>/dev/null; rm -rf "$LOCKDIR"' EXIT

# RTSP 密码从 secrets.env 的 LOREX_PASS 注入（避免明文入库）
if [ -z "${LOREX_PASS:-}" ]; then
  echo "$(date) ERROR: LOREX_PASS 未设置，请在 deploy/secrets.env 中配置" >> "$LOG"
  exit 1
fi
RTSP_URL="rtsp://admin:${LOREX_PASS}@192.168.2.15:554/cam/realmonitor?channel=1&subtype=0"
SEG=1800                 # 每段 30 分钟（连续切片，立即开录，无空窗）
FF='/usr/local/bin/ffmpeg'

while true; do
  ensure_dirs
  echo "$(date) start ffmpeg" >> "$LOG"
  "$FF" -rtsp_transport tcp -fflags +genpts \
    -i "$RTSP_URL" \
    -c:v copy -an -tag:v hvc1 \
    -f segment -segment_time "$SEG" -segment_wrap 0 -strftime 1 -reset_timestamps 1 \
    -segment_format mp4 \
    -segment_format_options movflags=+frag_keyframe+empty_moov+default_base_moof \
    -loglevel error \
    "$BASE/%Y/%m/%d/Garage_%Y%m%d_%H%M%S.mp4" >> "$LOG" 2>&1
  echo "$(date) ffmpeg exited($?) restart in 5s" >> "$LOG"
  sleep 5
done
