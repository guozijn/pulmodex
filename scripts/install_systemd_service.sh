#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="pulmodex.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_UID="$(id -u "${RUN_USER}")"
RUN_GID="$(id -g "${RUN_USER}")"

usage() {
  echo "Usage: $0 install|uninstall" >&2
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This script must be run as root. Use: sudo $0 $1" >&2
    exit 1
  fi
}

install_service() {
  require_root install
  mkdir -p "${APP_DIR}/outputs" "${APP_DIR}/checkpoints" "${APP_DIR}/uploads"
  chown "${RUN_USER}:$(id -gn "${RUN_USER}")" "${APP_DIR}/outputs" "${APP_DIR}/uploads"

  cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Pulmodex Docker Compose stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${APP_DIR}
User=${RUN_USER}
Group=$(id -gn "${RUN_USER}")
SupplementaryGroups=docker
Environment=HOST_UID=${RUN_UID}
Environment=HOST_GID=${RUN_GID}
Environment=COMPOSE_PROJECT_NAME=pulmodex
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

  sudo -u "${RUN_USER}" env HOST_UID="${RUN_UID}" HOST_GID="${RUN_GID}" docker compose --project-directory "${APP_DIR}" build
  systemctl daemon-reload
  systemctl enable "${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"
  systemctl --no-pager --full status "${SERVICE_NAME}" || true
}

uninstall_service() {
  require_root uninstall
  systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
  systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
  rm -f "${SERVICE_PATH}"
  systemctl daemon-reload
}

case "${1:-}" in
  install)
    install_service
    ;;
  uninstall)
    uninstall_service
    ;;
  *)
    usage
    exit 2
    ;;
esac
