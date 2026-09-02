#!/bin/sh
# 在 WDMyCloud 上安装 Lorex 录制：crontab 看门狗 + 30天清理 + 立即启动
BASE=/shares/Public/Lorex_Records

# 1) 写 crontab：每5分钟保活 + 每天清理30天前片段
( crontab -l 2>/dev/null | grep -v 'lorex_record' | grep -v 'Lorex_Records'; \
  echo "*/5 * * * * pgrep -f /root/lorex_record.sh >/dev/null 2>&1 || /root/lorex_record.sh >> $BASE/cron.log 2>&1"; \
  echo "17 3 * * * find $BASE -name 'Garage_*.mp4' -mtime +30 -delete" ) | crontab -
echo "== crontab =="
crontab -l

# 2) 确保 cron 守护进程在跑
pgrep cron >/dev/null 2>&1 || /usr/sbin/cron
sleep 1
echo "cron pid: $(pgrep cron)"

# 3) 立即启动录制（如未运行）
if ! pgrep -f /root/lorex_record.sh >/dev/null 2>&1; then
  nohup /root/lorex_record.sh >> $BASE/record.log 2>&1 &
  sleep 2
fi
echo "recorder pid: $(pgrep -f /root/lorex_record.sh)"
echo "done"
