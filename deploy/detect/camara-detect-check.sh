#!/bin/zsh
# 早间汇总：守护系统产出（raccoon 已禁用，未知动物统一走 animal 运动兜底）
# 设计定位（2026-09-03）：自建快照 = 事件时间定位器；具体事件以 Lorex app + 录像回放为准。
#   本脚本从 WD 上的 events CSV（由 lorex_detect.py 实时写入并上传）生成
#   「按类计数 + person 置信区间 + 可回放事件时间线」，时间线每行带 smb 录像路径与片段内偏移，
#   可直接在 Mac 打开 //192.168.2.90/Public/Lorex_Records/.../段.mp4 并 seek 到对应秒数核对。
# 触发时机：每天 09:00 EDT
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=15 -o HostKeyAlgorithms=+ssh-rsa,ssh-dss -o KexAlgorithms=+diffie-hellman-group1-sha1 -o Ciphers=+aes128-cbc,aes256-cbc,3des-cbc"
# 敏感凭证从同目录 secrets.env 读取（该文件已 gitignore，不入库）
if [ -f "$(dirname "$0")/secrets.env" ]; then
  source "$(dirname "$0")/secrets.env"
fi
sshpass -p "$WD_PASS" ssh $=O root@192.168.2.90 "$(cat <<'REMOTE'
TODAY=$(date +%Y-%m-%d)
YEST=$(date -d yesterday +%Y-%m-%d)
echo "=== 早间汇总 (触发 $TODAY 09:00 EDT) ==="
echo "窗口: $YEST 18:00 → $TODAY 08:59（事件时间线来自 events CSV）"
# 收集事件 CSV（昨天 + 今天，存在才纳入）
EV=""
for d in "$YEST" "$TODAY"; do
  p="/shares/Public/Lorex_Snapshots/events/$d.csv"
  [ -f "$p" ] && EV="$EV $p"
done
if [ -z "$EV" ]; then
  echo "  (无事件 CSV，可能守护尚未写入)"
else
  echo "--- 按类计数 (unknown_animal / raccoon 已归一为 animal) ---"
  awk -F, -v ys="$YEST 18:00:00" -v te="$TODAY 08:59:59" '
  NR==1 { next }
  {
    ts=$1
    if (ts>=ys && ts<=te) {
      c=$2; if (c=="unknown_animal" || c=="raccoon") c="animal"
      cnt[c]++
      if ($2=="person") {
        v=$3+0
        if (min=="" || v<min) min=v
        if (max=="" || v>max) max=v
      }
    }
  }
  END {
    for (k in cnt) printf "  %s: %d\n", k, cnt[k]
    printf "--- person conf: min=%s max=%s (基线 0.45-0.92)\n", (min==""?"NA":min), (max==""?"NA":max)
  }' $EV

  echo "--- 可回放事件时间线 (时间 | 类 | conf/area | 片段@偏移 | 录像smb) ---"
  awk -F, -v ys="$YEST 18:00:00" -v te="$TODAY 08:59:59" '
  NR==1 { next }
  {
    ts=$1
    if (ts>=ys && ts<=te) {
      c=$2; if (c=="unknown_animal" || c=="raccoon") c="animal"
      d=substr($5,8,8); yr=substr(d,1,4); mo=substr(d,5,2); dy=substr(d,7,2)
      rec="smb://192.168.2.90/Public/Lorex_Records/"yr"/"mo"/"dy"/"$5
      printf "%s | %s | %s | %s@%ss | %s\n", ts, c, $3, $5, $6, rec
    }
  }' $EV | sort
fi
echo "注: raccoon 已于 2026-09-03 禁用，未知动物统一走 animal 兜底；具体事件以 Lorex app + 录像回放为准。"
REMOTE
)"
