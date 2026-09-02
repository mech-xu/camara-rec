#!/bin/zsh
# 早间汇总：昨晚浣熊模型 + 守护系统产出
# 触发时机：每天 09:00 EDT
O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=15 -o HostKeyAlgorithms=+ssh-rsa,ssh-dss -o KexAlgorithms=+diffie-hellman-group1-sha1 -o Ciphers=+aes128-cbc,aes256-cbc,3des-cbc"
# 敏感凭证从同目录 secrets.env 读取（该文件已 gitignore，不入库）
if [ -f "$(dirname "$0")/secrets.env" ]; then
  source "$(dirname "$0")/secrets.env"
fi
sshpass -p "$WD_PASS" ssh $=O root@192.168.2.90 '
TODAY=$(date +%Y-%m-%d)
YEST=$(date -d yesterday +%Y-%m-%d)
TODAY_DIR=${TODAY:0:4}/${TODAY:5:2}/${TODAY:8:2}
YEST_DIR=${YEST:0:4}/${YEST:5:2}/${YEST:8:2}
echo "=== 早间汇总 (触发 $TODAY 09:00) ==="
echo "--- 昨晚 18:00-23:59 + 今日 00:00-08:59 快照按类计数 ---"
for day_dir in "$YEST_DIR" "$TODAY_DIR"; do
  for c in person cat dog raccoon unknown_animal; do
    n=$(find /shares/Public/Lorex_Snapshots/$day_dir -type f -name "*_${c}_*" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "$day_dir $c: $n"
  done
done
echo "--- 关键发现 ---"
echo "raccoon 数: $(find /shares/Public/Lorex_Snapshots/$YEST_DIR /shares/Public/Lorex_Snapshots/$TODAY_DIR -type f -name "*raccoon*" 2>/dev/null | wc -l)"
echo "unknown_animal 数: $(find /shares/Public/Lorex_Snapshots/$YEST_DIR /shares/Public/Lorex_Snapshots/$TODAY_DIR -type f -name "*unknown_animal*" 2>/dev/null | wc -l)"
echo "person 首位（最低 conf）: $(awk -F, "NR>1 && \$2==\"person\"{print \$3}" /shares/Public/Lorex_Snapshots/events/$YEST.csv /shares/Public/Lorex_Snapshots/events/$TODAY.csv 2>/dev/null | sort -n | head -1)"
echo "person 末位（最高 conf）: $(awk -F, "NR>1 && \$2==\"person\"{print \$3}" /shares/Public/Lorex_Snapshots/events/$YEST.csv /shares/Public/Lorex_Snapshots/events/$TODAY.csv 2>/dev/null | sort -n | tail -1)"
'
