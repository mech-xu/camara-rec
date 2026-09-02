#!/bin/sh
# 切换到「按录制时间归入 YYYY/MM/DD 子目录」布局，且不丢正在录的片段。
# 流程：等当前片段自然封口 -> 停录制 -> 删极小残片 -> 平铺片段按文件名日期迁入子目录 -> 以新布局重启
BASE=/shares/Public/Lorex_Records
LOG=$BASE/record.log
LOCKDIR=/tmp/lorex_record.lockdir

echo "$(date) [migrate] waiting for current segment to seal..." >> "$LOG"
CUR=$(ls -t "$BASE"/Garage_*.mp4 2>/dev/null | head -1)
i=0
while [ "$CUR" = "$(ls -t "$BASE"/Garage_*.mp4 2>/dev/null | head -1)" ] && [ $i -lt 400 ]; do
  sleep 5
  i=$((i+1))
done
echo "$(date) [migrate] new segment appeared (after ${i} polls) -> switching layout" >> "$LOG"

# 停录制（反斜杠点正则，避免误杀本脚本自身）
pkill -9 -f 'lorex_record\.sh' 2>/dev/null
pkill -9 ffmpeg 2>/dev/null
rm -rf "$LOCKDIR" 2>/dev/null
sleep 3

# 删掉刚开头几秒的极小残片（无 moov，不可播）
find "$BASE" -maxdepth 1 -name 'Garage_*.mp4' -size -2M -delete 2>/dev/null

# 其余平铺片段按文件名中的日期迁入 YYYY/MM/DD
for f in "$BASE"/Garage_*.mp4; do
  [ -e "$f" ] || continue
  n=$(basename "$f")
  d=$(echo "$n" | sed 's/Garage_\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)_.*/\1\/\2\/\3/')
  case "$d" in
    [0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9])
      if mkdir -p "$BASE/$d" && mv "$f" "$BASE/$d/"; then
        echo "$(date) [migrate] moved $n -> $d/" >> "$LOG"
      else
        echo "$(date) [migrate] FAILED to move $n" >> "$LOG"
      fi
      ;;
    *)
      echo "$(date) [migrate] skip (unparsable name): $n" >> "$LOG"
      ;;
  esac
done

# 以新布局启动
/etc/init.d/lorex-record start
echo "$(date) [migrate] done - restarted with dated-dir layout" >> "$LOG"
