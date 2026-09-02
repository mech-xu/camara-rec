#!/bin/sh
# 配置 WD 存活看门狗 + 过期片段清理(90天) + 空目录回收
# 清理周期依据：实测 236,189,429 B / 1800.99 s ≈ 1.049 Mbps
#   -> 472 MB/h -> 11.33 GB/天 -> 90 天约 1.02 TB（占 3.5TB 可用空间约 27%，留 2.5TB 余量）
# 看门狗调用 init.d（其内部 is_alive 自检，避开 pgrep 自匹配坑）
# 注意：过滤必须不区分大小写匹配 "lorex"，否则含连字符的 lorex-record 旧行不会被清掉 -> 重复
(crontab -l 2>/dev/null | grep -vi lorex) > /tmp/cron.new 2>/dev/null
cat >> /tmp/cron.new <<'EOF'
*/5 * * * * /etc/init.d/lorex-record start
17 3 * * * find /shares/Public/Lorex_Records -name 'Garage_*.mp4' -mtime +90 -delete
18 3 * * * find /shares/Public/Lorex_Records -mindepth 1 -type d -empty -delete
EOF
crontab /tmp/cron.new
echo "== 新 crontab =="
crontab -l
