#!/usr/bin/env bash
# releases/fitta_current 를 새 릴리스로 바꾸고, 재시작이 필요한지 알려준다.
#
# 화면 파일(web/static)만 바뀌었으면 재시작하지 않는다. uvicorn 이 정적 경로를
# 풀지 않고 들고 있어(app.py의 STATIC_DIR) 링크만 바꿔도 새 화면이 나간다.
# 파이썬 코드가 바뀌었으면 이미 적재된 모듈은 그대로이므로 반드시 재시작해야 한다.
#
# 재시작은 Slurm 작업을 놓고 다시 줄을 선다는 뜻이다. 우선순위가 대기 시간으로만
# 매겨지는 클러스터라 AGE 가 0 으로 초기화되고, 노드가 차 있으면 몇 시간씩 기다린다.
# 그래서 필요할 때만 재시작한다.
#
#   swap_release.sh <새 릴리스 폴더 이름> [--restart|--no-restart]
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

# 화면 파일만 바뀌었는지 본다. 그 외가 하나라도 있으면 재시작이 필요하다.
other=$(echo "$changed" | grep -v '^\s*$' | grep -v '^web/static/' || true)
if [ -n "$other" ]; then
  need_restart=yes
  echo "→ web/static 밖이 바뀌었습니다. 재시작이 필요합니다."
else
  need_restart=no
  echo "→ 화면 파일만 바뀌었습니다. 재시작 없이 반영됩니다."
fi

ln -sfn "$NEW" "$RELEASES/fitta_current"
echo "링크 전환 완료: $(readlink "$RELEASES/fitta_current")"

case "$MODE" in
  --restart) do_restart=yes ;;
  --no-restart) do_restart=no ;;
  *) do_restart=$need_restart ;;
esac

if [ "$do_restart" = yes ]; then
  echo "fitta-web 을 재시작합니다. GPU 를 놓고 다시 줄을 섭니다."
  systemctl --user restart fitta-web.service
  squeue -u "$USER" -n fitta-web -o "%.8i %.2t %R"
else
  echo "재시작하지 않았습니다. 실행 중인 작업이 그대로 새 화면을 서빙합니다."
fi
