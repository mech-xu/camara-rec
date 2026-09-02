#!/bin/sh
# 干净重启 Lorex 录制：杀掉所有旧实例 -> 清锁目录 -> 经 init.d 拉起单实例
# 本脚本自身 cmdline 为 /root/restart_recorder.sh，不含 lorex_record.sh，
# 故内部 pgrep -f "lorex_record.sh" 安全、不会误杀自己。
# 杀进程用 lorex_record\.sh（反斜杠点）：执行命令的 shell cmdline 含反斜杠，
# 正则不会匹配到自身，避免误杀。
echo "== stop old =="
pkill -9 -f 'lorex_record\.sh' 2>/dev/null
pkill -9 ffmpeg 2>/dev/null
rm -rf /tmp/lorex_record.lockdir 2>/dev/null
sleep 3
echo "after kill: loops=$(pgrep -f lorex_record.sh | wc -l) ffmpeg=$(pgrep ffmpeg | wc -l)"
echo "== start via init.d =="
/etc/init.d/lorex-record start
sleep 8
echo "after start: loops=$(pgrep -f lorex_record.sh | wc -l) ffmpeg=$(pgrep ffmpeg | wc -l)"
echo "lockdir present: $([ -d /tmp/lorex_record.lockdir ] && echo YES || echo NO)"
F=$(pgrep ffmpeg)
if [ -n "$F" ]; then
  echo "ffmpeg pid=$F ppid=$(awk '{print $4}' /proc/$F/stat) lockpid=$(cat /tmp/lorex_record.lockdir/pid 2>/dev/null)"
fi
N=$(ls -t /shares/Public/Lorex_Records/Garage_*.mp4 2>/dev/null | head -1)
echo "newest=$N"
S1=$(stat -c %s "$N")
sleep 4
S2=$(stat -c %s "$N")
echo "size $S1 -> $S2"
if [ "$S2" -gt "$S1" ]; then echo "growing=YES"; else echo "growing=NO"; fi
