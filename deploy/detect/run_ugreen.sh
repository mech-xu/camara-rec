#!/bin/sh
# Lorex 智能检测器 — UGREEN 守护入口
#  - 通过 setsid 脱离登录会话，避免 UGREEN 重启时随会话被杀
#  - 启动前先做一次 setup（挂载 WD、SSH 包装脚本）
#  - 崩溃则由 cron @reboot + 外部看门狗 5 分钟检查一次重启
export TZ=America/Toronto
export PYTHONUNBUFFERED=1
LOG=/home/sunny/camara-detect/daemon.log
mkdir -p /home/sunny/camara-detect

# 已运行则不再启动
if pgrep -f 'lorex_detect.py --daemon' >/dev/null 2>&1; then
  echo "$(date) 已在运行" >> "$LOG"
  exit 0
fi

# 必要时先 setup（挂载 WD sshfs、写 SSH 包装脚本）
/home/sunny/camara-detect/setup_ugreen.sh >> "$LOG" 2>&1

echo "$(date) 启动检测守护" >> "$LOG"
cd /home/sunny/camara-detect
exec setsid python3 /home/sunny/camara-detect/lorex_detect.py --daemon \
     >> "$LOG" 2>&1 < /dev/null
