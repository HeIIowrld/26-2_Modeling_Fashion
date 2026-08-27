const $ = (id) => document.getElementById(id);

/* 백엔드 주소. config.js가 window.FASHION_API_BASE를 정의한다.
   비어 있으면 같은 출처로 부른다(서버가 이 화면까지 함께 서빙하는 경우). */
const API_BASE = String(window.FASHION_API_BASE || "").replace(/\/+$/, "");

const state = {
  file: null,
  bodyFile: null,
  options: null,
  ruleTitles: {},
  jobId: null,
  result: null,
  step: 1,
  maxStep: 1,
  poll: null,
  retentionMinutes: 30,
  recommendations: [],
  profile: null,
  selected: 0,
  feedbackByRank: {},
  tryonByRank: {},
  tryon: { available: false, reason: "생성 모델을 준비 중입니다." },
  preferredColors: new Set(),
  avoidedColors: new Set(),
  preferredMaterials: new Set(),
  wardrobe: [],
};

const SCORE_LABELS = {
  purpose_tpo: "목적·격식",
  weather_activity: "날씨·활동",
  silhouette: "실루엣",
  color: "색상 조화",
  pattern_material_complexity: "패턴·소재",
  preference: "취향",
  wardrobe: "보유 옷",
};

const FIGURE_CAPTIONS = {
  original: "업로드한 원본 사진입니다.",
  landmarks: "MediaPipe가 찾은 어깨·골반·무릎·발목 관절입니다.",
  segmentation: "FASHN Human Parser가 나눈 의류 영역입니다.",
  preview: "VTON 미연결 상태의 추천 보드입니다. 실제 합성 결과가 아닙니다.",
};

/* ── 테마 ─────────────────────────────────────────────── */
const savedTheme = localStorage.getItem("fitta-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

$("theme-toggle").addEventListener("click", () => {
  const isDark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  const next = isDark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("fitta-theme", next);
});

/* ── 화면 전환 ────────────────────────────────────────── */
function goto(step) {
  if (step > state.maxStep) return;
  state.step = step;
  document.querySelectorAll(".panel").forEach((panel, index) => {
    panel.classList.toggle("is-active", index + 1 === step);
  });
  document.querySelectorAll(".stepper .step").forEach((el, index) => {
    const num = index + 1;
    const active = num === step;
    el.classList.toggle("is-active", active);
    el.classList.toggle("is-done", num < step);
    el.dataset.clickable = num <= state.maxStep ? "1" : "";
    el.toggleAttribute("disabled", num > state.maxStep);
    if (active) el.setAttribute("aria-current", "step");
    else el.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function unlock(step) {
  state.maxStep = Math.max(state.maxStep, step);
}

document.querySelectorAll(".stepper .step").forEach((el) => {
  el.addEventListener("click", () => goto(Number(el.dataset.goto)));
});

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, 2600);
}

/* ── 1단계: 업로드 ────────────────────────────────────── */
const dropzone = $("dropzone");
const fileInput = $("file-input");

dropzone.addEventListener("click", (event) => {
  if (event.target.id !== "clear-image") fileInput.click();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-drag");
  })
);
["dragleave", "drop"].forEach((name) =>
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-drag");
  })
);
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (file) acceptFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) acceptFile(fileInput.files[0]);
});
$("clear-image").addEventListener("click", (event) => {
  event.stopPropagation();
  state.file = null;
  fileInput.value = "";
  $("dropzone-preview").hidden = true;
  $("dropzone-empty").hidden = false;
  $("to-step-2").disabled = true;
  $("photo-gate-note").textContent = "사진을 선택하면 조건 입력으로 넘어갈 수 있어요.";
  $("photo-gate-note").dataset.tone = "wait";
});

function acceptFile(file) {
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
    toast("JPG, PNG, WEBP 이미지만 지원합니다.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    toast("이미지 용량은 12MB 이하만 지원합니다.");
    return;
  }
  state.file = file;
  $("preview-img").src = URL.createObjectURL(file);
  $("dropzone-empty").hidden = true;
  $("dropzone-preview").hidden = false;
  $("to-step-2").disabled = false;
  $("photo-gate-note").textContent = "사진이 준비됐어요. 다음 단계에서 원하는 코디 조건을 정해주세요.";
  $("photo-gate-note").dataset.tone = "ok";
}

/* 체형 파악용 사진 (선택) */
const bodyDropzone = $("body-dropzone");
const bodyFileInput = $("body-file-input");

bodyDropzone.addEventListener("click", (event) => {
  if (event.target.id !== "clear-body-image") bodyFileInput.click();
});
bodyDropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); bodyFileInput.click(); }
});
function acceptBodyFile(file) {
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) return toast("JPG, PNG, WEBP 이미지만 지원합니다.");
  if (file.size > 12 * 1024 * 1024) return toast("이미지 용량은 12MB 이하만 지원합니다.");
  state.bodyFile = file;
  $("body-preview-img").src = URL.createObjectURL(file);
  $("body-dz-empty").hidden = true;
  $("body-dz-preview").hidden = false;
}
bodyFileInput.addEventListener("change", () => {
  const file = bodyFileInput.files?.[0];
  if (!file) return;
  acceptBodyFile(file);
});
[["dragenter", true], ["dragover", true], ["dragleave", false], ["drop", false]].forEach(([name, active]) => {
  bodyDropzone.addEventListener(name, (event) => {
    event.preventDefault();
    bodyDropzone.classList.toggle("is-drag", active);
    if (name === "drop" && event.dataTransfer.files?.[0]) acceptBodyFile(event.dataTransfer.files[0]);
  });
});
$("clear-body-image").addEventListener("click", (event) => {
  event.stopPropagation();
  state.bodyFile = null;
  bodyFileInput.value = "";
  $("body-dz-preview").hidden = true;
  $("body-dz-empty").hidden = false;
});

$("to-step-2").addEventListener("click", () => { unlock(2); goto(2); });
$("back-to-1").addEventListener("click", () => goto(1));

/* ── 2단계: 조건 ──────────────────────────────────────── */
function fillSelect(id, values) {
  // 문자열 목록과 {value, label} 목록을 모두 받는다.
  const select = $(id);
  select.innerHTML = values
    .map((item) => (typeof item === "string" ? { value: item, label: item } : item))
    .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
    .join("");
}

function updateSelectionCount(id, store) {
  const counter = $(id);
  if (counter) counter.textContent = `${store.size}개 선택`;
}

function colorSwatches(container, store, counterId) {
  container.innerHTML = "";
  state.options.colors.forEach(({ name, rgb }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "swatch";
    button.innerHTML = `<i style="background: rgb(${rgb.join(",")})"></i>${name}`;
    button.addEventListener("click", () => {
      if (store.has(name)) store.delete(name);
      else store.add(name);
      button.classList.toggle("is-on", store.has(name));
      button.setAttribute("aria-pressed", String(store.has(name)));
      updateSelectionCount(counterId, store);
    });
    button.setAttribute("aria-pressed", "false");
    container.appendChild(button);
  });
}

function materialPills(container, store) {
  container.innerHTML = "";
  state.options.materials.forEach((item) => {
    const option = typeof item === "string" ? { value: item, label: item } : item;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill";
    button.textContent = option.label;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      if (store.has(option.value)) store.delete(option.value);
      else store.add(option.value);
      button.classList.toggle("is-on", store.has(option.value));
      button.setAttribute("aria-pressed", String(store.has(option.value)));
    });
    container.appendChild(button);
  });
}

// 예산은 최소·최대를 직접 입력한다. 슬라이더는 원하는 금액을 정확히 맞추기 어렵고,
// 10,000원 단위로만 끊겨 "23만원"처럼 실제로 생각하는 예산을 넣을 수 없었다.
const BUDGET_MIN = 30000;
const BUDGET_MAX = 500000;

// 입력 중에는 자릿수만 넣어준다. 커서가 끝으로 튀지 않도록 뒤쪽 길이를 기준으로 되돌린다.
function bindBudgetInput(el) {
  if (!el) return;
  el.addEventListener("input", () => {
    const tailBefore = el.value.length - (el.selectionStart ?? 0);
    const digits = el.value.replace(/[^0-9]/g, "").slice(0, 9);
    el.value = digits ? Number(digits).toLocaleString("ko-KR") : "";
    const caret = Math.max(0, el.value.length - tailBefore);
    el.setSelectionRange(caret, caret);
    el.setAttribute("aria-invalid", "false");
  });
  // 범위를 벗어나면 조용히 넘어가지 않고 되돌린 뒤 무엇을 고쳤는지 알린다.
  el.addEventListener("blur", () => {
    const value = Number(el.value.replace(/[^0-9]/g, "") || 0);
    if (!value) return;
    if (value < BUDGET_MIN) {
      el.value = BUDGET_MIN.toLocaleString("ko-KR");
      toast(`최소 ${BUDGET_MIN.toLocaleString("ko-KR")}원부터 추천할 수 있어 금액을 올렸어요.`);
    } else if (value > BUDGET_MAX) {
      el.value = BUDGET_MAX.toLocaleString("ko-KR");
      toast(`최대 ${BUDGET_MAX.toLocaleString("ko-KR")}원까지 받을 수 있어 금액을 낮췄어요.`);
    }
  });
}
bindBudgetInput($("f-min-budget"));
bindBudgetInput($("f-max-budget"));

$("add-wardrobe").addEventListener("click", () => addWardrobeRow());

function addWardrobeRow() {
  const list = $("wardrobe-list");
  const row = document.createElement("div");
  row.className = "wardrobe-row";
  row.innerHTML = `
    <select class="w-category"><option value="top">상의</option><option value="bottom">하의</option></select>
    <label class="wardrobe-upload">
      <input class="w-image" type="file" accept="image/jpeg,image/png,image/webp" hidden />
      <span class="wardrobe-upload-copy">옷 사진 선택</span>
      <img class="wardrobe-thumb" alt="보유 옷 미리보기" hidden />
    </label>
    <button class="row-remove" type="button" aria-label="삭제">×</button>`;
  const input = row.querySelector(".w-image");
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
      input.value = "";
      toast("보유 옷 사진은 JPG, PNG, WEBP만 지원합니다.");
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      input.value = "";
      toast("보유 옷 사진은 장당 12MB 이하만 지원합니다.");
      return;
    }
    row._file = file;
    const thumb = row.querySelector(".wardrobe-thumb");
    thumb.src = URL.createObjectURL(file);
    thumb.hidden = false;
    row.querySelector(".wardrobe-upload-copy").textContent = file.name;
    row.classList.add("has-image");
  });
  row.querySelector(".row-remove").addEventListener("click", () => row.remove());
  list.appendChild(row);
}

function wardrobeRowsWithImages() {
  return [...document.querySelectorAll(".wardrobe-row")].filter((row) => row._file);
}

function collectProfile() {
  const form = $("condition-form");
  const data = new FormData(form);
  const numeric = (key) => {
    const raw = data.get(key);
    if (raw === null || raw === "") return null;
    const cleaned = String(raw).replace(/[^0-9\-\.]/g, "");
    return cleaned === "" ? null : Number(cleaned);
  };
  const min_b = numeric("min_budget");
  const max_b = numeric("max_budget");
  const computeBudget = () => {
    if (min_b != null && max_b != null) return Math.round((min_b + max_b) / 2);
    const single = numeric("budget");
    return single != null ? single : null;
  };
  return {
    purpose: data.get("purpose"),
    desired_style: data.get("desired_style"),
    change_scope: data.get("change_scope"),
    season: data.get("season"),
    min_budget: min_b,
    max_budget: max_b,
    budget: computeBudget(),
    activity_level: data.get("activity_level"),
    height_cm: numeric("height_cm"),
    weight_kg: numeric("weight_kg"),
    preferred_colors: [...state.preferredColors],
    avoided_colors: [...state.avoidedColors],
    preferred_materials: [...state.preferredMaterials],
    excluded_item_types: [],
    owned_items: wardrobeRowsWithImages().map((row, index) => ({
      item_id: `OWN-${index + 1}`,
      category: row.querySelector(".w-category").value,
      image_index: index,
    })),
  };
}

/* ── 3단계: 분석 ──────────────────────────────────────── */
function renderStages(activeKey, doneKeys) {
  const list = $("stage-list");
  list.innerHTML = "";
  state.options.stages.forEach(({ key, label }) => {
    const li = document.createElement("li");
    li.className = "stage";
    if (doneKeys.includes(key)) li.classList.add("is-done");
    if (key === activeKey) li.classList.add("is-active");
    li.innerHTML = `<span class="stage-dot"></span><span>${label}</span>`;
    list.appendChild(li);
  });
}

function updateProgress(stageKey) {
  const keys = state.options.stages.map((s) => s.key);
  const index = keys.indexOf(stageKey);
  const done = index < 0 ? keys : keys.slice(0, index);
  renderStages(stageKey, done);
  const ratio = index < 0 ? 1 : index / keys.length;
  // width 가 아니라 transform 을 움직인다(레이아웃 재계산 없음). CSS 와 짝을 맞춰야 한다.
  $("progress-fill").style.transform = `scaleX(${Math.max(ratio, 0.06)})`;
}

$("start-analysis").addEventListener("click", async () => {
  if (!state.file) { toast("먼저 전신사진을 올려주세요."); goto(1); return; }
  const profileCandidate = collectProfile();
  // required fields: purpose, desired style, change scope, budget range
  if (!profileCandidate.purpose) { toast("코디 목적을 선택해주세요"); goto(2); return; }
  if (!profileCandidate.desired_style) { toast("원하는 스타일을 선택해주세요"); goto(2); return; }
  if (!profileCandidate.change_scope) { toast("바꾸고 싶은 범위를 선택해주세요"); goto(2); return; }
  if (profileCandidate.min_budget == null || profileCandidate.max_budget == null) { toast("예산 범위를 모두 입력해주세요"); goto(2); return; }
  if (profileCandidate.min_budget > profileCandidate.max_budget) { toast("최소 예산이 최대 예산보다 큽니다"); goto(2); return; }

  unlock(3);
  goto(3);
  $("error-card").hidden = true;
  $("progress-note").textContent = "모델을 준비하고 있어요. 첫 실행은 1분 이상 걸릴 수 있어요.";
  updateProgress(state.options.stages[0].key);

  const body = new FormData();
  body.append("image", state.file);
  state.profile = profileCandidate;
  body.append("profile", JSON.stringify(state.profile));
  if (state.bodyFile) body.append("body_image", state.bodyFile);
  wardrobeRowsWithImages().forEach((row) => body.append("wardrobe_images", row._file));

  try {
    const response = await fetch(API_BASE + "/api/analyze", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "분석 요청에 실패했어요.");
    state.jobId = payload.job_id;
    pollJob();
  } catch (error) {
    showError(error.message);
  }
});

function pollJob() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "상태를 확인할 수 없습니다.");
      if (payload.status === "running") {
        updateProgress(payload.stage);
        $("progress-note").textContent = "분석 중입니다. 창을 닫지 마세요.";
        return;
      }
      clearInterval(state.poll);
      if (payload.status === "failed") { showError(payload.error); return; }
      updateProgress(null);
      state.result = payload.result;
      renderResult(payload.result);
      unlock(4);
      goto(4);
    } catch (error) {
      clearInterval(state.poll);
      showError(error.message);
    }
  }, 1200);
}

function showError(message) {
  $("error-card").hidden = false;
  $("error-message").textContent = message;
  $("progress-note").textContent = "중단되었습니다.";
  $("progress-fill").style.transform = "scaleX(0)";
}

$("error-back").addEventListener("click", () => goto(2));

/* ── 4단계: 결과 ──────────────────────────────────────── */
function resetPrivacyBar() {
  const bar = $("privacy-bar");
  bar.classList.remove("is-deleted");
  bar.querySelector("strong").innerHTML =
    `업로드한 사진은 분석 완료 후 <span id="retention-minutes">${state.retentionMinutes}</span>분 이내 자동으로 삭제돼요.`;
  bar.querySelector("span:not(#retention-minutes)").textContent =
    "지금 바로 지우려면 오른쪽 버튼을 누르세요. 결과 이미지도 함께 사라져요.";
  $("delete-now").disabled = false;
}

function renderResult(result) {
  const isMock = Boolean(result?.mock);
  $("result-data-badge").hidden = !isMock;
  $("result-disclaimer").textContent = isMock
    ? "현재 분석 수치와 상품은 서비스 흐름을 확인하기 위한 시연 데이터예요. 실제 모델 서버를 연결하면 실제 분석 결과로 바뀝니다."
    : "추천 보드는 코디의 분위기를 확인하기 위한 참고 이미지이며 실제 핏을 보장하지 않습니다.";

  // Handle explicit budget-mismatch structured response from the server.
  if (result && result.budget_match === false) {
    $("result-lede").textContent = "착장 분석은 완료했지만 입력한 예산 안에서 추천 후보를 만들지 못했어요.";
    $("reco-count").textContent = "";
    $("budget-mismatch").hidden = false;
    // Hide the recommendation view so the user focuses on fixing budget.
    $("view-recos").hidden = true;
    $("tab-recos").disabled = true;
    // Button returns the user to page 2 (conditions) without clearing inputs.
    $("budget-fix-btn").onclick = () => {
      goto(2);
      $("budget-mismatch").hidden = true;
      $("view-recos").hidden = false;
      $("tab-recos").disabled = false;
    };

    // Still render analysis details so user can inspect outfit while deciding budget.
    resetPrivacyBar();
    renderCurrentOutfit(result);
    renderBodyStats(result.pose);
    renderRules(result.rules);
    renderFigures(result.images);
    showView("analysis");
    return;
  }

  const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
  const best = recommendations[0];
  resetPrivacyBar();
  if (!best) {
    $("result-lede").textContent = "조건에 맞는 추천 후보를 만들지 못했어요. 조건을 조금 넓혀 다시 시도해주세요.";
    $("reco-count").textContent = "";
    $("reco-picker").replaceChildren();
    $("reco-detail").innerHTML = '<div class="error-card"><h2>추천 후보가 없습니다</h2><p>예산이나 코디 조건을 조정한 뒤 다시 분석해주세요.</p></div>';
    renderCurrentOutfit(result);
    renderBodyStats(result.pose);
    renderRules(result.rules);
    renderFigures(result.images);
    showView("recos");
    return;
  }
  $("result-lede").textContent = best.products.length
    ? `점수가 가장 높은 코디부터 보여드립니다. 카드를 눌러 다른 후보를 확인하세요.`
    : "현재 코디를 유지하는 조건이라 보완 안내만 표시합니다.";
  $("reco-count").textContent = best.products.length ? recommendations.length : "";
  if (result.tryon) state.tryon = result.tryon;

  renderCurrentOutfit(result);
  renderBodyStats(result.pose);
  renderRecommendations(recommendations);
  renderRules(result.rules);
  renderFigures(result.images);
  showView("recos");
}

/* 상위 탭: 추천 / 분석 */
function showView(name) {
  document.querySelectorAll(".view-tab").forEach((tab) => {
    const on = tab.dataset.view === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
    view.classList.toggle("is-active", view.id === `view-${name}`);
  });
}

document.querySelectorAll(".view-tab").forEach((tab) => {
  tab.addEventListener("click", () => showView(tab.dataset.view));
});

function renderCurrentOutfit(result) {
  const { outfit_summary: summary, outfit } = result;
  $("current-outfit").innerHTML = `
    <div class="outfit-row">
      <span class="outfit-tag">상의</span>
      <div>
        <div class="outfit-desc">${escapeHtml(summary["상의"])}</div>
        <div class="outfit-meta">${escapeHtml(joinKnown([outfit.fit, outfit.neckline, outfit.material]))}</div>
      </div>
    </div>
    <div class="outfit-row">
      <span class="outfit-tag">하의</span>
      <div>
        <div class="outfit-desc">${escapeHtml(summary["하의"])}</div>
        <div class="outfit-meta">${escapeHtml(joinKnown([outfit.lower_fit, outfit.lower_material]))}</div>
      </div>
    </div>
    <div class="outfit-row">
      <span class="outfit-tag">조합</span>
      <div>
        <div class="outfit-desc">${escapeHtml(outfit.color_harmony)}</div>
        <div class="outfit-meta">${escapeHtml(joinKnown([outfit.style, outfit.silhouette]))}</div>
      </div>
    </div>`;

  const rows = [
    ["상의 종류", outfit.upper_type, outfit.attribute_sources.upper_type],
    ["레이어드", outfit.layering_state, outfit.attribute_sources.layering_state],
    ["감지된 상의", (outfit.upper_items || []).join(" + "), outfit.attribute_sources.layering_state],
    ["소매 길이", outfit.sleeve_length, outfit.attribute_sources.sleeve_length],
    ["현재 보이는 소매", outfit.visible_sleeve_length, outfit.attribute_sources.sleeve_state],
    ["소매 착용 상태", outfit.sleeve_state, outfit.attribute_sources.sleeve_state],
    ["소매 형태", outfit.sleeve_shape, outfit.attribute_sources.sleeve_shape],
    ["상의 기장", outfit.upper_length, outfit.attribute_sources.upper_length],
    ["넥라인", outfit.neckline, outfit.attribute_sources.neckline],
    ["칼라", outfit.collar, outfit.attribute_sources.collar],
    ["상의 핏", outfit.fit, outfit.attribute_sources.fit],
    ["상의 패턴", outfit.pattern, outfit.attribute_sources.pattern],
    ["상의 소재", outfit.material, outfit.attribute_sources.material],
    ["실루엣", outfit.silhouette, outfit.attribute_sources.silhouette],
    ["상의 디테일", outfit.details.join(", "), outfit.attribute_sources.details],
    ["하의 대분류", outfit.lower_type, outfit.attribute_sources.lower_type],
    ["하의 종류", outfit.lower_subtype, outfit.attribute_sources.lower_subtype],
    ["다리 모양", outfit.pant_leg_shape, outfit.attribute_sources.pant_leg_shape],
    ["바지 기장", outfit.pant_length, outfit.attribute_sources.pant_length],
    ["하의 핏", outfit.lower_fit, outfit.attribute_sources.lower_fit],
    ["하의 패턴", outfit.lower_pattern, outfit.attribute_sources.lower_pattern],
    ["하의 소재", outfit.lower_material, outfit.attribute_sources.lower_material],
    ["하의 디테일", outfit.lower_details.join(", "), outfit.attribute_sources.lower_details],
  ].filter(([, value]) => value);

  $("detail-table").innerHTML = rows
    .map(
      ([label, value, source]) => `
      <dl class="detail-row">
        <dt>${escapeHtml(label)}</dt>
        <dd>${escapeHtml(value)}</dd>
        ${source ? `<span class="src-tag">${escapeHtml(sourceLabel(source))}</span>` : "<span></span>"}
      </dl>`
    )
    .join("");
}

$("toggle-detail").addEventListener("click", () => {
  const table = $("detail-table");
  table.hidden = !table.hidden;
  $("toggle-detail").textContent = table.hidden ? "상세 보기" : "상세 닫기";
});

function sourceLabel(source) {
  return {
    trained_head: "학습 헤드",
    trained_lower_detail_head: "학습 헤드",
    fused_agreement: "합의",
    zero_shot: "제로샷",
    mask: "마스크 측정",
    derived_category: "카테고리 유도",
    derived_category_collar: "칼라 유도",
    derived_attribute_conflict: "속성 교차검증",
    zero_shot_roi: "부위별 제로샷",
    visible_mask_and_trained_head: "학습 헤드+마스크",
    not_visible: "가려짐",
  }[source] || source;
}

function renderBodyStats(pose) {
  const confidence = pose.body_shape_confidence;
  const level = confidence >= 0.8 ? "high" : confidence >= 0.65 ? "mid" : "low";
  $("body-stats").innerHTML = `
    <dl class="stat-row">
      <dt>체형 참고 분류</dt>
      <dd>${escapeHtml(pose.body_shape)}
        <span class="confidence-pill ${level}">${(confidence * 100).toFixed(0)}%</span>
      </dd>
    </dl>
    <dl class="stat-row"><dt>어깨·골반 비율</dt><dd>${pose.shoulder_hip_ratio.toFixed(2)}</dd></dl>
    <dl class="stat-row"><dt>상·하체 비율</dt><dd>${pose.upper_lower_ratio.toFixed(2)}</dd></dl>
    <dl class="stat-row"><dt>다리 길이 비율</dt><dd>${pose.leg_ratio.toFixed(2)}</dd></dl>
    <dl class="stat-row"><dt>자세</dt><dd>${escapeHtml(pose.posture)}</dd></dl>`;

  // 체형을 어떻게 다루는지는 라벨을 실제로 보는 이 자리에서 설명해요.
  const goal = state.profile?.silhouette_goal;
  const basis = pose.body_shape_basis === "입력한 둘레"
    ? "입력하신 둘레로 판정했어요."
    : "사진에서 추정했어요. 둘레를 입력하면 더 세분화돼요.";
  const used = goal && goal !== "별도 보정 없음"
    ? `'${goal}'를 고르셔서 추천 점수에 반영했어요.`
    : "체형은 추천 근거를 설명하는 참고 정보로만 보여드려요.";
  $("body-shape-note").textContent = `${basis} ${used}`;
}

function renderRecommendations(recommendations) {
  state.recommendations = recommendations;
  state.feedbackByRank = {};
  state.tryonByRank = {};

  const picker = $("reco-picker");
  picker.innerHTML = "";
  // 후보가 하나뿐이면(현재 코디 유지) 고를 것이 없어 선택 카드를 감춘다.
  picker.hidden = recommendations.length < 2;

  recommendations.forEach((reco, index) => {
    const total = reco.products.reduce((sum, product) => sum + product.price, 0);
    const card = document.createElement("button");
    card.type = "button";
    card.className = `pick${index === 0 ? " is-on" : ""}`;
    card.setAttribute("role", "tab");
    card.setAttribute("aria-selected", String(index === 0));
    card.innerHTML = `
      <div class="pick-top">
        <span class="pick-rank"><span class="rank-badge">${reco.rank}</span>순위</span>
        <span class="pick-score">${reco.total_score.toFixed(1)}</span>
      </div>
      <div class="pick-swatches">
        ${reco.products.map((p) => `<i style="background: rgb(${p.color_rgb.join(",")})"></i>`).join("")}
      </div>
      <div class="pick-names">${reco.products.map((p) => escapeHtml(p.name)).join("<br />")}</div>
      <div class="pick-price">${total.toLocaleString("ko-KR")}원</div>`;
    card.addEventListener("click", () => selectRecommendation(index));
    picker.appendChild(card);
  });

  selectRecommendation(0);
}

function selectRecommendation(index) {
  const reco = state.recommendations[index];
  if (!reco) return;
  state.selected = index;

  document.querySelectorAll(".pick").forEach((card, position) => {
    card.classList.toggle("is-on", position === index);
    card.setAttribute("aria-selected", String(position === index));
  });

  const total = reco.products.reduce((sum, product) => sum + product.price, 0);
  const products = reco.products
    .map(
      (product) => `
      <div class="product">
        <span class="product-swatch" style="background: rgb(${product.color_rgb.join(",")})"></span>
        <div>
          <div class="product-name">${escapeHtml(product.name)}</div>
          <div class="product-meta">${escapeHtml(joinKnown([product.item_type, product.fit, product.material, product.style]))}</div>
        </div>
        <span class="product-price">${product.price.toLocaleString("ko-KR")}원</span>
      </div>`
    )
    .join("");

  const bars = Object.entries(reco.score_breakdown)
    .map(
      ([key, value]) => `
      <div class="score-bar">
        <span>${escapeHtml(SCORE_LABELS[key] || key)}</span>
        <span class="score-track"><span class="score-value" style="transform: scaleX(${Math.min(value, 100) / 100})"></span></span>
        <b>${value.toFixed(0)}</b>
      </div>`
    )
    .join("");

  const head = reco.products.length
    ? `<span class="reco-rank"><span class="rank-badge">${reco.rank}</span>추천 코디</span>
       <span class="reco-score">${reco.total_score.toFixed(1)}<small>점 · ${total.toLocaleString("ko-KR")}원</small></span>`
    : `<span class="reco-rank"><span class="rank-badge">＝</span>현재 코디 유지</span>
       <span class="card-note">새로 구매할 상품 없음</span>`;

  const chosen = state.feedbackByRank[reco.rank];

  $("reco-detail").innerHTML = `
    <article class="reco is-top">
      <div class="reco-head">${head}</div>
      <div class="reco-body">
        ${reco.products.length ? tryonBlock(reco) : ""}
        ${products ? `<div class="product-list">${products}</div>` : ""}
        ${reco.reasons.length ? `<ul class="reasons">${reco.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>` : ""}
        ${
          reco.styling_tips.length
            ? `<div class="tips-block"><h3>스타일링 팁</h3><ul>${reco.styling_tips
                .map((tip) => `<li>${escapeHtml(tip)}</li>`)
                .join("")}</ul></div>`
            : ""
        }
        ${
          bars
            ? `<details class="mini-disclosure"><summary>점수 상세 보기</summary>
                 <div class="score-bars">${bars}</div>
                 ${
                   reco.applied_rules.length
                     ? `<div class="rule-chips">${reco.applied_rules
                         .map((id) => `<span class="rule-chip" title="${escapeHtml(state.ruleTitles[id] || id)}">${id}</span>`)
                         .join("")}</div>`
                     : ""
                 }
               </details>`
            : ""
        }
        <div class="reco-foot">
          <div class="feedback-group" data-rank="${reco.rank}">
            ${["마음에 들어요", "별로예요", "저장"]
              .map(
                (action) =>
                  `<button class="fb-btn${chosen === action ? " is-on" : ""}" type="button" data-action="${action}">${action}</button>`
              )
              .join("")}
          </div>
          <span class="coverage-note">${reco.products.length ? `계산 범위 ${reco.score_coverage.toFixed(0)}%` : ""}</span>
        </div>
      </div>
    </article>`;

  $("reco-detail").querySelector(".feedback-group").addEventListener("click", sendFeedback);
  const tryonButton = $("reco-detail").querySelector(".tryon-btn");
  if (tryonButton) tryonButton.addEventListener("click", () => requestTryon(reco.rank));
}

/* 예상 착장샷 — 생성 모델이 붙기 전에는 자리만 잡아 둔다. */
function tryonBlock(reco) {
  const done = state.tryonByRank[reco.rank];
  if (done) {
    return `
      <div class="tryon is-done">
        <img src="${API_BASE}/api/jobs/${state.jobId}/images/${done}" alt="추천 코디 예상 착장샷" />
        <p class="tryon-note">${state.result?.mock ? "서비스 흐름 확인용 시연 이미지입니다." : "생성 모델이 만든 예상 이미지입니다. 실제 핏을 보장하지 않습니다."}</p>
      </div>`;
  }
  const ready = state.tryon.available;
  return `
    <div class="tryon${ready ? "" : " is-pending"}">
      <div class="tryon-slot">
        <div class="tryon-icon" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.4"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 8l6 4 6-4 8 5-3 7-3-1v19H16V19l-3 1-3-7z" />
          </svg>
        </div>
        <div class="tryon-copy">
          <strong>예상 착장샷</strong>
          <span>${escapeHtml(ready ? "이 코디를 입은 모습을 생성해 봅니다." : state.tryon.reason)}</span>
        </div>
        <button class="btn btn-ghost btn-sm tryon-btn" type="button" ${ready ? "" : "disabled"}>
          ${ready ? "생성하기" : "준비 중"}
        </button>
      </div>
    </div>`;
}

async function requestTryon(rank) {
  const box = $("reco-detail").querySelector(".tryon");
  const button = box.querySelector(".tryon-btn");
  button.disabled = true;
  button.textContent = "생성 중…";
  box.classList.add("is-working");
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}/tryon/${rank}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "예상 착장샷을 만들지 못했습니다.");
    state.tryonByRank[rank] = payload.image;
    selectRecommendation(state.selected);
  } catch (error) {
    button.disabled = false;
    button.textContent = "다시 시도";
    box.classList.remove("is-working");
    toast(error.message);
  }
}

async function sendFeedback(event) {
  const button = event.target.closest(".fb-btn");
  if (!button) return;
  const group = event.currentTarget;
  const rank = Number(group.dataset.rank);
  try {
    const response = await fetch(API_BASE + "/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rank, action: button.dataset.action }),
    });
    if (!response.ok) throw new Error("피드백을 저장하지 못했습니다.");
    // 다른 후보를 봤다가 돌아와도 선택이 남아 있도록 기억한다.
    state.feedbackByRank[rank] = button.dataset.action;
    group.querySelectorAll(".fb-btn").forEach((el) => el.classList.remove("is-on"));
    button.classList.add("is-on");
    toast("피드백을 저장했습니다.");
  } catch (error) {
    toast(error.message);
  }
}

function renderRules(rules) {
  $("rule-summary").innerHTML = `
    <div class="rule-metric"><span>구현된 규칙</span><b>${rules.implemented} / ${rules.documented}</b></div>
    <div class="rule-metric"><span>순위 점수 규칙</span><b>${rules.scoring}</b></div>
    <div class="rule-metric"><span>추가 데이터 필요</span><b>${rules.unsupported.length}</b></div>`;
  $("unsupported-list").innerHTML = rules.unsupported
    .map((item) => `<li><code>${escapeHtml(item.id)}</code> — ${escapeHtml(item.reason)}</li>`)
    .join("");
}

function renderFigures(images) {
  const tabs = $("figure-tabs");
  tabs.querySelectorAll(".figure-tab").forEach((tab) => {
    tab.onclick = () => {
      tabs.querySelectorAll(".figure-tab").forEach((el) => el.classList.remove("is-active"));
      tab.classList.add("is-active");
      showFigure(images, tab.dataset.image);
    };
  });
  tabs.querySelectorAll(".figure-tab").forEach((el, index) => el.classList.toggle("is-active", index === 0));
  showFigure(images, "original");
}

function showFigure(images, key) {
  $("figure-img").src = `${API_BASE}/api/jobs/${state.jobId}/images/${images[key]}`;
  const mockCaptions = {
    original: "업로드한 원본 사진입니다.",
    landmarks: "시연용 관절 위치 화면입니다.",
    segmentation: "시연용 의류 분리 화면입니다.",
    preview: "추천 코디의 분위기를 정리한 시연용 보드입니다.",
  };
  $("figure-caption").textContent = (state.result?.mock ? mockCaptions : FIGURE_CAPTIONS)[key] || "";
}

$("delete-now").addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("삭제하지 못했습니다.");
    const bar = $("privacy-bar");
    bar.classList.add("is-deleted");
    bar.querySelector("strong").textContent = "사진과 결과 이미지를 삭제했습니다.";
    bar.querySelector("span").textContent = "화면에 남은 분석 내용은 새로고침하면 사라집니다.";
    $("delete-now").disabled = true;
    document.querySelectorAll(".figure img").forEach((img) => img.removeAttribute("src"));
    $("figure-caption").textContent = "삭제되었습니다.";
    $("clear-image").click();
    $("clear-body-image").click();
    $("wardrobe-list").replaceChildren();
    state.jobId = null;
    toast("사진을 삭제했습니다.");
  } catch (error) {
    toast(error.message);
  }
});

$("restart").addEventListener("click", () => {
  $("clear-image").click();
  $("clear-body-image").click();
  $("wardrobe-list").replaceChildren();
  state.result = null;
  state.jobId = null;
  state.maxStep = 1;
  goto(1);
});
$("tweak").addEventListener("click", () => goto(2));

/* ── 유틸 ─────────────────────────────────────────────── */
function joinKnown(values) {
  const blocked = ["분석 보류", "분석 불가", "불확실", "해당 없음", ""];
  return values.filter((v) => v && !blocked.some((b) => b && v.includes(b))).join(" · ");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch])
  );
}

/* ── 초기화 ───────────────────────────────────────────── */
function showBackendDown(detail) {
  /* 정적 호스팅(GitHub Pages 등)에 화면만 올라가 있고 연산 서버가 꺼져 있는 상태.
     조용히 실패하면 버튼만 안 먹는 것처럼 보여서, 이유를 화면에 띄운다. */
  const banner = document.createElement("div");
  banner.className = "backend-down";
  banner.innerHTML =
    "<strong>분석 서버에 연결하지 못했습니다.</strong>" +
    "<span>이 화면은 정적 호스팅에 올라간 사본입니다. 분석·합성은 별도 서버가 켜져 있어야 동작합니다.</span>" +
    (API_BASE
      ? `<code>${escapeHtml(API_BASE)}</code>`
      : "<code>API 주소가 설정되지 않았습니다 (config.js)</code>");
  document.body.prepend(banner);
  console.error("백엔드 연결 실패:", detail);
}

function showPreviewOnly(detail) {
  /* 정적 호스팅에서도 fallback 선택지로 화면을 그대로 보여준다.
     백엔드 연결 상태는 방문자 화면에 노출하지 않고 개발자 콘솔에만 남긴다. */
  console.warn("분석 서버 없음 — 정적 선택지로 화면만 표시:", detail);
}

(async function init() {
  let options;
  try {
    const response = await fetch(API_BASE + "/api/options");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    options = await response.json();
  } catch (error) {
    /* 정적 호스팅(GitHub Pages 등)에는 연산 서버가 없다. 예전에는 여기서 바로
       빠져나가서 목적·예산 같은 선택지가 통째로 비어 있었고, 그래서 디자인을
       검토하려고 들어온 사람이 조건 화면에서 아무것도 볼 수 없었다.
       같은 폴더에 떠 둔 실제 선택지로 화면을 채우고 연결 실패는 콘솔에 남긴다. */
    try {
      const fallback = await fetch("fallback-options.json");
      if (!fallback.ok) throw new Error(`HTTP ${fallback.status}`);
      options = await fallback.json();
    } catch {
      showBackendDown(error);
      return;
    }
    showPreviewOnly(error);
  }
  state.options = options;

  fillSelect("f-purpose", options.purposes);
  fillSelect("f-style", options.styles);
  fillSelect("f-scope", options.change_scopes);
  fillSelect("f-season", options.seasons);
  fillSelect("f-activity", options.activity_levels);
  // Do not pre-select required fields. Insert placeholder for required selects.
  ["f-purpose", "f-style", "f-scope"].forEach((id) => {
    const sel = $(id);
    if (sel) {
      sel.insertAdjacentHTML('afterbegin', '<option value="">선택해주세요</option>');
      sel.value = "";
    }
  });
  $("f-season").value = "자동";
  $("f-activity").value = "보통";

  colorSwatches($("preferred-colors"), state.preferredColors, "preferred-color-count");
  colorSwatches($("avoided-colors"), state.avoidedColors, "avoided-color-count");
  materialPills($("preferred-materials"), state.preferredMaterials);
  renderStages(null, []);

  fetch(API_BASE + "/api/retention")
    .then((response) => response.json())
    .then((policy) => {
      state.retentionMinutes = policy.ttl_minutes;
      document.querySelectorAll("#retention-minutes").forEach((el) => {
        el.textContent = policy.ttl_minutes;
      });
      const hint = document.querySelector(".dz-privacy");
      if (hint) hint.textContent = `사진은 분석 완료 후 ${policy.ttl_minutes}분 이내 자동 삭제`;
    })
    .catch(() => {});

  fetch(API_BASE + "/api/rules")
    .then((response) => response.json())
    .then((payload) => { state.ruleTitles = payload.titles || {}; })
    .catch(() => {});

  // health info suppressed from topbar in this UI
  // fetch(API_BASE + "/api/health").catch(() => {});
})();
