#!/usr/bin/env bash
# 실행 중인 서버는 확정된 릴리스 경로를 사용하므로 화면만 바뀌어도 재시작한다.
# --no-restart 는 링크만 준비하고 실제 반영을 미루는 명시적 옵션이다.
# swap_release.sh <새 릴리스 폴더 이름> [--restart|--no-restart]
set -uo pipefail

RELEASES=/data1/dsl01/releases
NEW=${1:?사용법: swap_release.sh <릴리스 폴더 이름> [--restart|--no-restart]}
MODE=${2:-auto}

[ -d "$RELEASES/$NEW" ] || { echo "없는 릴리스: $RELEASES/$NEW" >&2; exit 1; }
CURRENT=$(readlink "$RELEASES/fitta_current" || true)
[ -n "$CURRENT" ] || { echo "fitta_current 링크를 읽지 못했습니다." >&2; exit 1; }
CURRENT=${CURRENT##*/}
if [ "$CURRENT" = "$NEW" ]; then
  echo "이미 $NEW 를 가리키고 있습니다."
  exit 0
fi

echo "현재: $CURRENT"
echo "대상: $NEW"

changed=$(diff -rq --strip-trailing-cr \
  --exclude=__pycache__ --exclude='*.pyc' --exclude=REVISION \
  "$RELEASES/$CURRENT" "$RELEASES/$NEW" 2>/dev/null \
  | sed -e "s#^Files $RELEASES/$CURRENT/##" -e "s# and .*##" \
        -e "s#^Only in $RELEASES/$CURRENT: #(삭제) #" \
        -e "s#^Only in $RELEASES/$NEW: #(추가) #")

if [ -z "$changed" ]; then
  echo "코드 차이가 없습니다."
else
  echo "바뀐 파일:"
  echo "$changed" | sed 's/^/  /'
fi

# 확정 경로로 제출된 실행/대기 작업은 링크 변경을 자동으로 읽지 않는다.
need_restart=yes
ln -sfn "$NEW" "$RELEASES/fitta_current"
echo "링크 전환 완료: $(readlink "$RELEASES/fitta_current")"

case "$MODE" in
  --restart) do_restart=yes ;;
  --no-restart) do_restart=no ;;
  *) do_restart=$need_restart ;;
esac

if [ "$do_restart" = yes ]; then
  echo "fitta-web 을 재시작합니다. GPU 를 놓고 다시 줄을 섭니다."
  systemctl --user daemon-reload
  systemctl --user restart fitta-web.service
  sleep 3
  squeue -u "$USER" -n fitta-web -o "%.8i %.2t %.12l %R"
elif [ "$need_restart" = yes ]; then
  echo "링크만 바꿨습니다. 실행/대기 작업에는 새 릴리스가 아직 반영되지 않았습니다."
fi
