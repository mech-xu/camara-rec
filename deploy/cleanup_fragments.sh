#!/bin/sh
# 清理测试期间产生的「不可播残片」（旧格式 MP4 被中断，无 moov atom）
# 只按精确文件名删除，且跳过当前正在写入的片段。
DIR=/shares/Public/Lorex_Records/2026/08/29
echo "== 删除不可播残片 =="
for n in 153930 155436 155817 155914; do
  f=$DIR/Garage_20260829_$n.mp4
  if [ -f "$f" ]; then
    sz=$(stat -c %s "$f")
    if rm -f "$f"; then
      echo "  已删  Garage_20260829_$n.mp4  ($sz 字节)"
    else
      echo "  删除失败 Garage_20260829_$n.mp4"
    fi
  else
    echo "  不存在(跳过) Garage_20260829_$n.mp4"
  fi
done
echo "== 删除后清单 =="
ls -la $DIR/Garage_*.mp4
echo "== 剩余总数 =="
ls -1 $DIR/Garage_*.mp4 | wc -l
echo "== 录制状态(应不受影响) =="
echo "  ffmpeg=$(pgrep ffmpeg | wc -l) 监督循环相关进程=$(pgrep -f 'lorex_record\.sh' | wc -l)"
N=$(ls -t $DIR/Garage_*.mp4 | head -1)
S1=$(stat -c %s "$N")
sleep 5
S2=$(stat -c %s "$N")
echo "  当前片段 $(basename $N) 5秒增长=$((S2-S1)) 字节"
