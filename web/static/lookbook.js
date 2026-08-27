/* ============================================================
   룩북 화면 전용 동작. app.js 다음에 실행돼서 필요한 부분만 덮어쓴다.
   app.js 는 기존 화면과 공유하므로 한 줄도 고치지 않는다.

   두 가지를 바꾼다.
   1) 스테퍼를 4칸에서 3칸으로 — 분석은 사용자가 할 일이 없는 대기 상태라
      단계로 세면 "아직 절반"이라는 잘못된 인상을 준다. 결과 자리에 얹는다.
   2) 2단계에도 1단계와 같은 잠금 규칙 — 필수를 채우기 전에는 버튼이 잠기고,
      무엇이 남았는지 미리 알려준다. 눌러본 뒤에야 알게 두지 않는다.
   ============================================================ */
(() => {
  "use strict";

  /* ── 1. 3칸 스테퍼에 4개 패널 얹기 ─────────────────────── */

  // 패널 번호 → 스테퍼 칸 번호. 분석(3)은 결과(3번 칸)에서 진행 중으로 보인다.
  const slotOf = (step) => (step >= 3 ? 3 : step);

  // app.js 의 goto 를 같은 이름으로 다시 선언한다. app.js 안의 호출부는
  // 이름으로 찾아 쓰기 때문에 호출 시점에 이쪽이 잡힌다.
  window.goto = function goto(step) {
    if (step > state.maxStep) return;
    state.step = step;

    document.querySelectorAll(".panel").forEach((panel, index) => {
      panel.classList.toggle("is-active", index + 1 === step);
    });

    const slot = slotOf(step);
    document.querySelectorAll(".stepper .step").forEach((el, index) => {
      const pos = index + 1;
      const active = pos === slot;
      const clickable = Number(el.dataset.goto) <= state.maxStep;
      el.classList.toggle("is-active", active);
      el.classList.toggle("is-done", pos < slot);
      // 결과 칸은 분석 중일 때만 돌아가는 표시를 단다.
      el.classList.toggle("is-working", pos === 3 && step === 3);
      el.dataset.clickable = clickable ? "1" : "";
      el.toggleAttribute("disabled", !clickable);
      if (active) el.setAttribute("aria-current", "step");
      else el.removeAttribute("aria-current");
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  /* ── 2. 2단계 잠금 ────────────────────────────────────── */

  const REQUIRED = [
    ["f-purpose", "코디 목적"],
    ["f-style", "원하는 스타일"],
    ["f-scope", "바꾸고 싶은 범위"],
    ["f-min-budget", "최소 예산"],
    ["f-max-budget", "최대 예산"],
  ];

  const digits = (value) => String(value || "").replace(/[^0-9]/g, "");

  function missing() {
    return REQUIRED.filter(([id]) => {
      const el = document.getElementById(id);
      if (!el) return false;
      return el.tagName === "SELECT" ? !el.value : !digits(el.value);
    }).map(([, label]) => label);
  }

  function budgetOrderProblem() {
    const min = Number(digits(document.getElementById("f-min-budget").value) || 0);
    const max = Number(digits(document.getElementById("f-max-budget").value) || 0);
    return min && max && min > max ? "최소 예산이 최대 예산보다 큽니다" : null;
  }

  function refreshGate() {
    const button = document.getElementById("start-analysis");
    const note = document.getElementById("gate-note");
    if (!button || !note) return;

    const left = missing();
    const order = left.length ? null : budgetOrderProblem();
    const blocked = left.length > 0 || order !== null;

    button.disabled = blocked;
    if (order) {
      note.textContent = order;
      note.dataset.tone = "bad";
    } else if (left.length) {
      note.textContent = `남은 필수 항목: ${left.join(" · ")}`;
      note.dataset.tone = "wait";
    } else {
      note.textContent = "필요한 항목을 모두 채웠어요.";
      note.dataset.tone = "ok";
    }
  }

  const form = document.getElementById("condition-form");
  if (form) {
    form.addEventListener("input", refreshGate);
    form.addEventListener("change", refreshGate);
  }

  // 셀렉트는 서버에서 목록을 받아온 뒤에 채워진다. 그 전에 한 번,
  // 채워진 뒤에 다시 한 번 판정해야 버튼 상태가 실제와 어긋나지 않는다.
  refreshGate();
  const purpose = document.getElementById("f-purpose");
  if (purpose) {
    new MutationObserver(refreshGate).observe(purpose, { childList: true });
  }
  window.addEventListener("load", refreshGate);
})();
