#!/bin/sh
# 在 UGREEN 上准备 Lorex 智能检测环境
#   - 用 sshfs 以「只读」方式挂载 WD 共享（避免误删录像）
#   - 快照与事件日志通过 smbclient 回写（已验证匿名可写）
# 注意：sshfs 的 -o ssh_command= 里不能带空格/换行（会被 FUSE 误解析），
#       因此把老算法 SSH 参数包进一个独立脚本，再让 ssh_command 指向它。
set -e

LOCAL=/home/sunny/camara-detect
MOUNT=/home/sunny/wd_public      # /mnt 需 root，普通用户挂到 home 下
WD=192.168.2.90

mkdir -p "$LOCAL/events" /tmp/camara_detect
mkdir -p "$MOUNT"

# SSH 包装脚本：兼容 WD 的老算法（UGREEN→WD 已配好免密公钥）
cat > "$LOCAL/ssh_wd.sh" <<'EOF'
#!/bin/sh
exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o PubkeyAcceptedKeyTypes=+ssh-rsa -o HostKeyAlgorithms=+ssh-rsa,ssh-dss -o KexAlgorithms=+diffie-hellman-group1-sha1,diffie-hellman-group14-sha1 -o Ciphers=+aes128-cbc,aes256-cbc,3des-cbc "$@"
EOF
chmod +x "$LOCAL/ssh_wd.sh"

if mount | grep -q " $MOUNT "; then
  echo "== WD 已挂载 =="
else
  echo "== 挂载 WD (只读 sshfs) =="
  sshfs -o ro,reconnect,ServerAliveInterval=15,ssh_command="$LOCAL/ssh_wd.sh" \
        root@$WD:/shares/Public "$MOUNT" 2>&1 | head -5
  sleep 2
fi

echo "== 挂载校验 =="
mount | grep " $MOUNT " || { echo "挂载失败"; exit 1; }
echo "  Public 下："; ls "$MOUNT" | head -6
echo "  Lorex_Records 下："; ls "$MOUNT/Lorex_Records" | head -5

echo
echo "== 录像文件清单(最新 5 个) =="
find "$MOUNT/Lorex_Records" -name "Garage_*.mp4" 2>/dev/null | sort | tail -5
echo
echo "== 模型 =="
ls -la "$LOCAL/yolov8n.onnx" 2>/dev/null || echo "  模型缺失"
