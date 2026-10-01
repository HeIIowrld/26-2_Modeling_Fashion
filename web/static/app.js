const $ = (id) => document.getElementById(id);

/* 백엔드 주소. config.js가 window.FASHION_API_BASE를 정의한다.
   비어 있으면 같은 출처로 부른다(서버가 이 화면까지 함께 서빙하는 경우). */
const API_BASE = String(window.FASHION_API_BASE || "").replace(/\/+$/, "");

const state = {
  file: null,
  bodyFile: null,
  options: null,
  jobId: null,
  result: null,
  step: 1,
  maxStep: 1,
  poll: null,
  retentionMinutes: 30,
  profile: null,
  shoppingProducts: [],
  shoppingOutfits: [],
  lookSelected: 0,
  shoppingTryonResults: [],
  shoppingTryonBatch: null,
  shoppingTryonPoll: null,
  tryon: { available: false, reason: "생성 모델을 준비 중입니다." },
  preferredColors: new Set(),
  avoidedColors: new Set(),
  preferredMaterials: new Set(),
  wardrobe: [],
  photoValidation: false,
  photoWarningFile: null,
};

const BUDGET_MIN = 30000;
const BUDGET_MAX = 500000;
const BUDGET_STEP = 10000;

function syncBudgetSlider(changed) {
  const minInput = $("f-min-budget");
  const maxInput = $("f-max-budget");
  let minimum = Number(minInput.value);
  let maximum = Number(maxInput.value);
  if (minimum > maximum - BUDGET_STEP) {
    if (changed === minInput) minimum = maximum - BUDGET_STEP;
    else maximum = minimum + BUDGET_STEP;
  }
  minimum = Math.max(BUDGET_MIN, minimum);
  maximum = Math.min(BUDGET_MAX, maximum);
  minInput.value = String(minimum);
  maxInput.value = String(maximum);
  $("min-budget-output").textContent = `${minimum.toLocaleString("ko-KR")}원`;
  $("max-budget-output").textContent = `${maximum.toLocaleString("ko-KR")}원`;
  minInput.setAttribute("aria-valuetext", `${minimum.toLocaleString("ko-KR")}원`);
  maxInput.setAttribute("aria-valuetext", `${maximum.toLocaleString("ko-KR")}원`);
  const left = ((minimum - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN)) * 100;
  const right = 100 - ((maximum - BUDGET_MIN) / (BUDGET_MAX - BUDGET_MIN)) * 100;
  $("budget-selected").style.left = `${left}%`;
  $("budget-selected").style.right = `${right}%`;
  minInput.style.zIndex = minimum > BUDGET_MAX - BUDGET_STEP * 3 ? "4" : "3";
  maxInput.style.zIndex = "3";
}

const FIGURE_CAPTIONS = {
  original: "업로드한 원본 사진입니다.",
  landmarks: "MediaPipe가 찾은 어깨·골반·무릎·발목 관절입니다.",
  segmentation: "FASHN Human Parser가 나눈 의류 영역입니다.",
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
  state.photoValidation = false;
  state.photoWarningFile = null;
  fileInput.value = "";
  $("dropzone-preview").hidden = true;
  $("dropzone-empty").hidden = false;
  $("to-step-2").disabled = true;
  $("photo-gate-note").textContent = "사진을 선택하면 조건 입력으로 넘어갈 수 있어요.";
  $("photo-gate-note").dataset.tone = "wait";
});

/* 같은 아이폰 사진도 macOS·브라우저 조합에 따라 image/heic 대신
   image/heic-sequence 같은 MIME으로 전달될 수 있다. MIME 또는 확장자 중
   하나가 맞으면 업로드하고, 실제 파일 형식은 서버가 다시 검증한다. */
const IMAGE_TYPES = /^image\/(jpeg|jpg|pjpeg|png|webp|heic|heif|heic-sequence|heif-sequence|avif)$/i;
const IMAGE_EXTENSIONS = /\.(jpe?g|png|webp|heic|heif|avif)$/i;
const IMAGE_HINT = "JPG, PNG, WEBP, HEIC, AVIF 이미지만 지원합니다.";

function isSupportedImage(file) {
  return IMAGE_TYPES.test(file.type || "") || IMAGE_EXTENSIONS.test(file.name || "");
}

function acceptFile(file) {
  if (!isSupportedImage(file)) {
    toast(IMAGE_HINT);
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    toast("이미지 용량은 12MB 이하만 지원합니다.");
    return;
  }
  state.file = file;
  state.photoValidation = false;
  state.photoWarningFile = null;
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
  if (!isSupportedImage(file)) return toast(IMAGE_HINT);
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

async function validatePhotoBeforeNext() {
  if (!state.file || state.photoValidation) return;
  if (state.photoWarningFile === state.file) {
    state.photoWarningFile = null;
    unlock(2);
    goto(2);
    return;
  }
  const file = state.file;
  const button = $("to-step-2");
  const note = $("photo-gate-note");
  state.photoValidation = true;
  button.disabled = true;
  button.textContent = "사진을 확인하고 있어요";
  note.textContent = "사진을 확인하고 있어요. 잠시만 기다려 주세요.";
  note.dataset.tone = "wait";
  try {
    const body = new FormData();
    body.append("image", file);
    const response = await fetch(API_BASE + "/api/validate-photo", { method: "POST", body });
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error("서버 응답을 읽을 수 없습니다.");
    }
    if (state.file !== file) return;
    if (!response.ok) throw new Error(payload.detail || "사진 확인에 실패했어요. 잠시 후 다시 시도해 주세요.");
    const gateMessage = (messages) => (messages || [])
      .map((message) => String(message).trim().replace(/([.!?。])\s+/g, "$1\n"))
      .filter(Boolean)
      .join("\n");
    if (!payload.valid) {
      state.photoWarningFile = null;
      note.textContent = gateMessage(payload.issues) || "사진에서 사람의 정면 전신을 확인할 수 없습니다.\n다시 촬영해 주세요.";
      note.dataset.tone = "bad";
      return;
    }
    // 통과 경고를 1페이지에서 읽을 수 있게 한 번 멈춘다.
    // 사용자가 확인 버튼을 다시 누르면 조건 페이지로 이동한다.
    const warnings = payload.warnings || [];
    if (warnings.length) {
      note.textContent = gateMessage(warnings);
      note.dataset.tone = "warn";
      state.photoWarningFile = file;
      return;
    }
    note.textContent = "사진이 확인됐어요. 조건을 입력해주세요.";
    note.dataset.tone = "ok";
    unlock(2);
    goto(2);
  } catch (error) {
    note.textContent = error.message || "사진 확인 중 서버 오류가 발생했어요. 잠시 후 다시 시도해 주세요.";
    note.dataset.tone = "error";
  } finally {
    state.photoValidation = false;
    button.textContent = state.photoWarningFile === file
      ? "경고 확인 후 조건 입력"
      : "다음: 조건 입력";
    button.disabled = !state.file;
  }
}

$("to-step-2").addEventListener("click", validatePhotoBeforeNext);
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

$("add-wardrobe").addEventListener("click", () => addWardrobeRow());

function addWardrobeRow() {
  const list = $("wardrobe-list");
  const row = document.createElement("div");
  row.className = "wardrobe-row";
  row.innerHTML = `
    <select class="w-category"><option value="top">상의</option><option value="bottom">하의</option></select>
    <label class="wardrobe-upload">
      <input class="w-image" type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" hidden />
      <span class="wardrobe-upload-copy">옷 사진 선택</span>
      <img class="wardrobe-thumb" alt="보유 옷 미리보기" hidden />
    </label>
    <button class="row-remove" type="button" aria-label="삭제">×</button>`;
  const input = row.querySelector(".w-image");
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    if (!isSupportedImage(file)) {
      input.value = "";
      toast(`보유 옷 사진은 ${IMAGE_HINT}`);
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
    gender: data.get("gender"),
    desired_style: data.get("desired_style"),
    change_categories: data.getAll("change_categories"),
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
  const ratio = index < 0 ? 1 : (index + 1) / keys.length;
  // width 가 아니라 transform 을 움직인다(레이아웃 재계산 없음). CSS 와 짝을 맞춰야 한다.
  $("progress-fill").style.transform = `scaleX(${Math.max(ratio, 0.06)})`;
  $("progress-note").textContent = index >= 0
    ? `${state.options.stages[index].label} 중이에요…`
    : "분석과 생성을 모두 마쳤어요.";
}

$("start-analysis").addEventListener("click", async () => {
  if (!state.file) { toast("먼저 전신사진을 올려주세요."); goto(1); return; }
  const profileCandidate = collectProfile();
  // required fields: gender, purpose, desired style, change scope, budget range
  if (!profileCandidate.gender) { toast("성별을 선택해주세요"); goto(2); return; }
  if (!profileCandidate.purpose) { toast("코디 목적을 선택해주세요"); goto(2); return; }
  if (!profileCandidate.desired_style) { toast("원하는 스타일을 선택해주세요"); goto(2); return; }
  if (!profileCandidate.change_categories.length) { toast("바꾸고 싶은 부분을 하나 이상 선택해주세요"); goto(2); return; }
  if (profileCandidate.min_budget == null || profileCandidate.max_budget == null) { toast("예산 범위를 선택해주세요"); goto(2); return; }
  if (profileCandidate.min_budget > profileCandidate.max_budget) { toast("최소 예산이 최대 예산보다 큽니다"); goto(2); return; }

  unlock(3);
  stopShoppingTryonBatchPolling();
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
        return;
      }
      clearInterval(state.poll);
      if (payload.status === "failed") { showError(payload.error); return; }
      updateProgress(null);
      state.result = payload.result;
      renderResult(payload.result, payload.shopping_tryon_batch);
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

function renderRequestSummary(request) {
  const summary = $("request-summary");
  const values = request || state.profile;
  if (!values) { summary.hidden = true; return; }
  const chips = [
    ["성별", values.gender],
    ["목적", values.purpose],
    ["스타일", values.desired_style],
    ["변경", values.change_categories ? values.change_categories.map((category) => ({top: "상의", bottom: "하의", shoes: "신발"})[category]).join(" · ") : values.change_scope],
    ["예산", values.min_budget != null && values.max_budget != null
      ? `${Number(values.min_budget).toLocaleString("ko-KR")}~${Number(values.max_budget).toLocaleString("ko-KR")}원`
      : ""],
    ["계절", values.season],
    ["활동", values.activity_level],
    ["선호색", (values.preferred_colors || []).join("·")],
    ["제외색", (values.avoided_colors || []).join("·")],
    ["선호소재", (values.preferred_materials || []).join("·")],
  ].filter(([, value]) => value && value !== "자동");
  $("request-chips").innerHTML = chips
    .map(([label, value]) => `<span class="request-chip"><b>${escapeHtml(label)}</b> ${escapeHtml(value)}</span>`)
    .join("");
  summary.hidden = chips.length === 0;
}

function renderResult(result, initialBatch = null) {
  stopShoppingTryonBatchPolling();
  state.shoppingProducts = [];
  state.shoppingOutfits = [];
  state.shoppingTryonResults = [];
  state.shoppingTryonBatch = null;
  state.lookSelected = 0;
  $("shopping-tryon-panel").hidden = true;
  if (result.tryon) state.tryon = result.tryon;
  renderRequestSummary(result?.request);
  const isMock = Boolean(result?.mock);
  $("result-data-badge").hidden = !isMock;
  $("result-disclaimer").textContent = isMock
    ? "현재 분석 수치와 상품은 서비스 흐름을 확인하기 위한 시연 데이터예요. 실제 모델 서버를 연결하면 실제 분석 결과로 바뀝니다."
    : "무신사 상품의 가격과 재고는 상품 페이지에서 확인해주세요. 예상 착장샷은 실제 핏을 보장하지 않습니다.";

  const shoppingResults = Array.isArray(result.shopping_results) ? result.shopping_results : [];
  resetPrivacyBar();
  const shoppingOutfits = Array.isArray(result.shopping_outfits) ? result.shopping_outfits : null;
  $("result-lede").textContent = shoppingOutfits && !shoppingOutfits.length
    ? "저득점 조합을 제외한 뒤 안전선을 통과한 자동 코디가 없어 개별 상품만 보여드려요."
    : `현재 유지할 옷과 교체할 상품의 조화를 확인해 ${shoppingOutfits?.length || 1}가지 코디로 구성했습니다.`;

  renderCurrentOutfitEvaluation(result.current_outfit_evaluation);
  renderCurrentOutfit(result);
  renderBodyStats(result.pose);
  renderShoppingProducts(result.shopping_results || [], result.shopping_outfits || []);
  if (initialBatch?.items?.length) {
    applyShoppingTryonBatch(initialBatch);
  } else if (state.tryon.available
      && (result.shopping_results || []).some((product) => product.tryon_available)
      && (shoppingOutfits === null || shoppingOutfits.length)) {
    startShoppingTryonBatch();
  }
  renderFigures(result.images);
  showView("recos");
}

function renderSizeFit(fit) {
  /* 기준 옷 입력을 없앴으므로(2026-10-01, 팀 결정) 비교는 하지 않는다.
     상품이 표기한 실측 자체는 참조값과 무관하게 쓸모가 있어 표만 남긴다. */
  const columns = Object.entries(fit?.columns || {});
  const rows = fit?.size_options || [];
  if (!rows.length || !columns.length) return "";
  const cells = (measurements) => columns.map(([key]) =>
    `<td>${measurements?.[key] == null ? "—" : escapeHtml(String(measurements[key]))}</td>`
  ).join("");
  return `<section class="shopping-size-fit" aria-label="상품 실측">
    <details><summary>사이즈별 실측 보기 (cm)</summary>
      <div class="size-table-scroll"><table>
        <caption class="sr-only">상품 사이즈별 실측, 단위 cm</caption>
        <thead><tr><th scope="col">사이즈</th>${columns.map(([, label]) => `<th scope="col">${escapeHtml(label)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr><th scope="row">${escapeHtml(row.size)}</th>${cells(row.measurements)}</tr>`).join("")}</tbody>
      </table></div>
      <small>무신사 상품 표기 기준이에요. 소재와 신축성에 따라 착용감은 달라질 수 있어요.</small>
    </details>
  </section>`;
}

function renderShoppingProductCard(product) {
  const reviews = product.review_count
    ? `리뷰 ${Number(product.review_count).toLocaleString("ko-KR")}`
    : "";
  const rating = product.review_score
    ? `${(Number(product.review_score) / 20).toFixed(1)} / 5`
    : "";
  return `<article class="shopping-card">
    <a class="shopping-link" href="${escapeHtml(product.url)}" target="_blank"
       rel="noopener noreferrer sponsored" aria-label="무신사에서 ${escapeHtml(product.name)} 보기">
      <div class="shopping-image-wrap">
        <img class="shopping-image" src="${escapeHtml(product.image_url)}"
             alt="${escapeHtml(product.name)} 상품 사진" loading="lazy" referrerpolicy="no-referrer" />
        <span class="shopping-category">${({top: "상의", bottom: "하의", shoes: "신발"})[product.category] || "미지원"}</span>
      </div>
      <div class="shopping-copy">
        <span class="shopping-brand">${escapeHtml(product.brand || "MUSINSA")}</span>
        <h3>${escapeHtml(product.name)}</h3>
        <div class="shopping-meta">${escapeHtml([rating, reviews, product.gender].filter(Boolean).join(" · "))}</div>
        ${(product.search_keywords || []).length ? `
          <div class="shopping-keywords" aria-label="대표 검색 키워드">
            ${(product.search_keywords || []).slice(0, 3).map((keyword) =>
              `<span class="shopping-keyword">${escapeHtml(keyword)}</span>`
            ).join("")}
          </div>` : ""}
        ${renderShoppingReason(product)}
        <div class="shopping-bottom">
          <strong>${Number(product.price).toLocaleString("ko-KR")}원</strong>
          <span>무신사에서 보기 ↗</span>
        </div>
      </div>
    </a>
    ${renderSizeFit(product.size_fit)}
    ${product.source === "musinsa_catalog_fallback" ? `<small>${product.stock_checked_at ? `${escapeHtml(new Date(product.stock_checked_at).toLocaleDateString("ko-KR"))} 수집 기준` : "재고 확인 시점 미상"} · 구매 전 상품 페이지에서 가격과 재고를 확인해주세요.</small>` : ""}
  </article>`;
}

const CATEGORY_LABEL = { top: "상의", bottom: "하의", shoes: "신발" };

function combinationKey(productIds) {
  return [...productIds].sort().join("|");
}

function outfitEntry(outfit, index) {
  return {
    label: `LOOK ${index + 1}`,
    key: combinationKey((outfit.products || []).map((product) => product.product_id)),
  };
}

function lookEntries() {
  return state.shoppingOutfits.map(outfitEntry);
}

function outfitTryonResult(key) {
  return state.shoppingTryonResults.find(
    (result) => combinationKey((result.key || "").split("|")) === key
  ) || null;
}

function outfitBatchItem(key) {
  return (state.shoppingTryonBatch?.items || []).find(
    (item) => combinationKey(item.product_ids || []) === key
  ) || null;
}

/* 상품명을 ' + '로 이어 붙이면 긴 무신사 이름 세 개가 한 덩어리로 흘러 읽히지 않는다
   ("…레귤러 셔츠_블랙 + [시누 PICK] 아르코 … + SP2604 …").
   부위별로 한 줄씩 끊고 라벨을 붙인다. */
function lookItemList(result) {
  const names = result.names || [];
  const categories = result.categories || [];
  if (!names.length) return "";
  const rows = names.map((name, index) => ({
    label: CATEGORY_LABEL[categories[index]] || "",
    name,
  }));
  // 부위를 모르면 라벨 없이 줄만 나눈다. 순서는 상의 → 하의 → 신발.
  const order = ["상의", "하의", "신발", ""];
  rows.sort((a, b) => order.indexOf(a.label) - order.indexOf(b.label));
  return `<ul class="look-items">${rows.map((row) =>
    `<li>${row.label ? `<b>${escapeHtml(row.label)}</b>` : ""}<span>${escapeHtml(row.name)}</span></li>`
  ).join("")}</ul>`;
}

function renderOutfitTryon(entry) {
  const result = outfitTryonResult(entry.key);
  if (result) {
    const extension = result.image?.toLowerCase().endsWith(".png") ? "png" : "jpg";
    const notes = [...new Set(result.warnings || [])];
    const warnings = notes.length
      ? `<details class="tryon-warning">
          <summary>이 사진에서 참고할 점 ${notes.length}가지</summary>
          <ul>${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
        </details>`
      : "";
    return `<figure class="outfit-tryon-result">
        <img src="${API_BASE}/api/jobs/${state.jobId}/images/${result.image}"
             alt="${escapeHtml(entry.label)}을 적용한 예상 착장샷" />
        <figcaption>
          ${lookItemList(result)}
          <a href="${API_BASE}/api/jobs/${state.jobId}/images/${result.image}"
             download="fitta-${escapeHtml(entry.label)}.${extension}">사진 저장</a>
        </figcaption>
      </figure>${warnings}`;
  }
  const item = outfitBatchItem(entry.key);
  const status = item?.status || "";
  const reason = status === "failed" ? item?.error : "";
  const copy = reason || {
    queued: "합성 차례를 기다리는 중입니다.",
    running: "이 코디를 자동으로 합성하고 있습니다.",
    failed: "이 코디는 합성하지 못했습니다.",
  }[status] || (state.tryon.available
    ? "추천이 끝나면 이 자리에 예상 착장샷이 자동으로 나타납니다."
    : state.tryon.reason);
  return `<div class="outfit-tryon-result is-pending" data-status="${escapeHtml(status)}">
      <p>${escapeHtml(copy)}</p>
    </div>`;
}

function renderLookPanel(outfit, index) {
  const entry = outfitEntry(outfit, index);
  const currentItems = (outfit.current_items || []).map((item) => `
    <div class="outfit-current-item">
      <span>${escapeHtml(item.label || "현재 착장")}</span>
      <strong>${escapeHtml(item.description || "사진 속 아이템")}</strong>
    </div>`).join("");
  const evidence = (outfit.evidence || []).slice(0, 3);
  const labels = outfit.evidence_labels || [];
  /* 규칙 문장일 때 outfit.reason 은 근거 앞의 두 문장을 이어 붙인 값이라, 요약과
     근거 펼침에 같은 문장이 두 번 보였다. 근거를 라벨과 함께 바로 보이고,
     LLM 이 근거를 합쳐 다듬은 문장이 있을 때만 그 문장을 대신 보인다. */
  const reasons = outfit.reason_source === "llm" || !evidence.length
    ? `<p class="look-summary">${sentenceLines(outfit.reason
        || "현재 착장과 선택 조건을 함께 고려한 조합입니다.")}</p>`
    : `<ul class="look-reasons">${evidence.map((text, evidenceIndex) => `<li>${labels[evidenceIndex] ? `<b>${escapeHtml(labels[evidenceIndex])}</b>` : ""}<span>${escapeHtml(text)}</span></li>`).join("")}</ul>`;
  return `<section class="look-panel" role="tabpanel" id="look-panel-${index}"
      aria-labelledby="look-tab-${index}" tabindex="0">
    ${reasons}
    ${currentItems ? `<div class="outfit-current-items" aria-label="그대로 입는 현재 아이템">${currentItems}</div>` : ""}
    <div class="look-body">
      <div class="look-render" data-outfit-render="${escapeHtml(entry.key)}">
        ${renderOutfitTryon(entry)}
      </div>
      <div class="look-products">
        ${(outfit.products || []).map(renderShoppingProductCard).join("")}
      </div>
    </div>
  </section>`;
}

function renderLooks(entries) {
  const index = Math.min(state.lookSelected, entries.length - 1);
  state.lookSelected = Math.max(0, index);
  const statusLabel = { queued: "대기", running: "생성 중", done: "완료", failed: "실패" };
  const tabs = entries.map((entry, position) => {
    const result = outfitTryonResult(entry.key);
    const status = result ? "done" : (outfitBatchItem(entry.key)?.status || "");
    return `<button type="button" role="tab" id="look-tab-${position}"
      aria-controls="look-panel-${position}" aria-selected="${position === state.lookSelected}"
      tabindex="${position === state.lookSelected ? 0 : -1}" data-look-tab="${position}"
      class="${position === state.lookSelected ? "is-on" : ""}">
      ${escapeHtml(entry.label)}${status ? `<small data-look-status="${escapeHtml(status)}">${escapeHtml(statusLabel[status] || status)}</small>` : ""}
    </button>`;
  }).join("");
  return `<div class="look-tabs" role="tablist" aria-label="추천 코디 조합">${tabs}</div>
    ${renderLookPanel(state.shoppingOutfits[state.lookSelected], state.lookSelected)}`;
}

function renderShoppingProducts(products, outfits = []) {
  const section = $("shopping-section");
  const grid = $("shopping-results");
  const panel = $("shopping-tryon-panel");
  if (!section || !grid) return;
  if (!products.length) {
    section.hidden = false;
    grid.innerHTML = '<div class="error-card"><h2>검색 결과가 없습니다</h2><p>예산이나 선호 조건을 조금 넓혀 다시 검색해주세요.</p></div>';
    if (panel) panel.hidden = true;
    return;
  }
  state.shoppingProducts = products;
  state.shoppingOutfits = outfits;
  grid.classList.toggle("has-outfit-combinations", Boolean(outfits.length));
  grid.innerHTML = outfits.length
    ? renderLooks(lookEntries())
    : state.shoppingProducts.map((product, index) => `
        ${index === 0 || state.shoppingProducts[index - 1].category !== product.category
          ? `<h3 class="shopping-category-heading">${({top: "상의", bottom: "하의", shoes: "신발"})[product.category] || "상품"} 추천 · ${state.shoppingProducts.filter((item) => item.category === product.category).length}개</h3>` : ""}
        ${renderShoppingProductCard(product)}`).join("");
  const requested = state.result?.request?.change_categories || [];
  const noSafeOutfits = Array.isArray(state.result?.shopping_outfits) && !outfits.length;
  const shortages = outfits.length
    ? (outfits.length < 3 ? requested : [])
    : requested.filter((category) => products.filter((product) => product.category === category).length < 3);
  if (shortages.length) {
    grid.insertAdjacentHTML("beforeend", `<p class="shopping-category-heading">조건에 맞는 코디 조합이 3개보다 적습니다. 예산·조건을 조정해 다시 검색해보세요.</p>`);
  }
  if (noSafeOutfits) {
    grid.insertAdjacentHTML("beforeend", '<p class="shopping-category-heading">저득점 조합은 추천에서 제외했습니다. 조건을 조정하거나 상품을 직접 골라 입어볼 수 있어요.</p>');
  }
  const lookTabs = [...grid.querySelectorAll("[data-look-tab]")];
  const selectLook = (index, moveFocus) => {
    state.lookSelected = (index + lookTabs.length) % lookTabs.length;
    renderShoppingProducts(state.shoppingProducts, state.shoppingOutfits);
    if (moveFocus) grid.querySelector(`[data-look-tab="${state.lookSelected}"]`)?.focus();
  };
  lookTabs.forEach((button) => {
    button.addEventListener("click", () => selectLook(Number(button.dataset.lookTab), false));
    button.addEventListener("keydown", (event) => {
      const current = Number(button.dataset.lookTab);
      const moves = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: lookTabs.length - 1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      selectLook(moves[event.key], true);
    });
  });
  renderShoppingTryonPanel();
  section.hidden = false;
}

/* 규칙 문장일 때 recommendation_reason 은 근거 목록의 첫 문장을 그대로 복사한 값이라,
   요약과 목록을 함께 보이면 같은 문장이 두 번 나왔다. 근거 목록을 라벨과 함께 한 칸에
   보이고, LLM 이 근거를 합쳐 다듬은 문장이 있을 때만 그 문장을 대신 보인다. */
function renderShoppingReason(product) {
  const evidence = (product.fit_evidence || []).slice(0, 3);
  const labels = product.fit_evidence_labels || [];
  const ruleIds = (product.reason_rule_ids || []).join(" ");
  const useSentence = product.recommendation_reason_source === "llm" || !evidence.length;
  if (useSentence && !product.recommendation_reason) return "";
  const body = useSentence
    ? `<span>${escapeHtml(product.recommendation_reason)}</span>`
    : `<ul>${evidence.map((text, index) => `<li>${labels[index] ? `<b>${escapeHtml(labels[index])}</b>` : ""}<span>${escapeHtml(text)}</span></li>`).join("")}</ul>`;
  return `<div class="shopping-reason" data-reason-rule-ids="${escapeHtml(ruleIds)}">
    <b>상품 선택 근거</b>${body}
  </div>`;
}

function refreshOutfitTryonRenders() {
  const slots = new Map(
    [...document.querySelectorAll("[data-outfit-render]")]
      .map((slot) => [slot.dataset.outfitRender, slot])
  );
  state.shoppingOutfits.forEach((outfit, index) => {
    const entry = outfitEntry(outfit, index);
    const slot = slots.get(entry.key);
    if (slot) slot.innerHTML = renderOutfitTryon(entry);
  });
  const entries = lookEntries();
  const statusLabel = { queued: "대기", running: "생성 중", done: "완료", failed: "실패" };
  document.querySelectorAll("[data-look-tab]").forEach((button) => {
    const entry = entries[Number(button.dataset.lookTab)];
    const status = outfitTryonResult(entry.key)
      ? "done" : (outfitBatchItem(entry.key)?.status || "");
    const badge = button.querySelector("small");
    if (!status) return badge?.remove();
    const text = statusLabel[status] || status;
    if (badge) {
      badge.dataset.lookStatus = status;
      badge.textContent = text;
    } else {
      button.insertAdjacentHTML("beforeend", `<small data-look-status="${escapeHtml(status)}">${escapeHtml(text)}</small>`);
    }
  });
}

function renderShoppingTryonPanel() {
  const panel = $("shopping-tryon-panel");
  if (!panel) return;
  const batch = state.shoppingTryonBatch;
  const batchLabels = {
    idle: "조합 계산 중",
    queued: "전체 조합 렌더 대기",
    running: "전체 조합 렌더링 중",
    done: "전체 조합 렌더링 완료",
    partial: "일부 조합 렌더링 완료",
    failed: "조합 렌더링 실패",
    unavailable: "전체 조합 렌더링 불가",
  };
  const batchProgress = batch
    ? `<div class="shopping-batch" data-status="${escapeHtml(batch.status)}">
         <div class="shopping-batch-copy">
           <strong>${escapeHtml(batchLabels[batch.status] || "추천 코디 조합")}</strong>
           <span>${Number(batch.ready || 0)} / ${Number(batch.total || 0)}장 준비</span>
         </div>
         ${batch.total ? `<div class="shopping-batch-progress" aria-label="무신사 조합 렌더링 진행률">
           <span style="transform:scaleX(${Math.min(1, Number(batch.finished || 0) / Number(batch.total || 1))})"></span>
         </div>
         <div class="shopping-batch-items">${(batch.items || []).map((item) => `
           <span class="is-${escapeHtml(item.status)}">조합 ${item.index} · ${escapeHtml({queued:"대기",running:"생성 중",done:"완료",failed:"실패"}[item.status] || item.status)}</span>`).join("")}</div>` : ""}
         ${batch.reason ? `<p>${escapeHtml(batch.reason)}</p>` : ""}
       </div>`
    : "";
  // 사진에서 하의 밑단이 잘리면 기장 차이를 잴 수 없다. 추측하지 않고 사용자에게 묻는다.
  const lengthCheck = state.result?.length_check;
  const lengthPrompt = lengthCheck && ["needs_input", "user_input"].includes(lengthCheck.status)
    ? `<div class="length-check" data-status="${escapeHtml(lengthCheck.status)}">
         <strong>지금 입은 하의 기장</strong>
         <p>${escapeHtml(lengthCheck.message)}</p>
         <div class="pills" role="group" aria-label="지금 입은 하의 기장">
           ${(lengthCheck.options || []).map((option) => `
             <button type="button" class="pill ${lengthCheck.value === option ? "is-on" : ""}"
               data-bottom-length="${escapeHtml(option)}"
               aria-pressed="${lengthCheck.value === option}">${escapeHtml(option)}</button>`).join("")}
         </div>
       </div>`
    : "";
  panel.innerHTML = `${lengthPrompt}${batchProgress}`;
  panel.hidden = !(lengthPrompt || batchProgress);
  panel.querySelectorAll("[data-bottom-length]").forEach((button) => {
    button.addEventListener("click", () => submitCurrentBottomLength(button.dataset.bottomLength));
  });
}

async function submitCurrentBottomLength(length) {
  if (!state.jobId) return;
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}/current-bottom-length`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ length }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "하의 기장을 반영하지 못했습니다.");
    state.result.length_check = payload.length_check;
    // 이미 끝난 조합의 기장 경고도 새 기준으로 바뀌므로 수동 결과까지 함께 맞춘다.
    const refreshed = new Map((payload.shopping_tryon_batch?.items || [])
      .filter((item) => item.status === "done")
      .map((item) => [(item.product_ids || []).join("|"), item.warnings || []]));
    state.shoppingTryonResults.forEach((result) => {
      if (refreshed.has(result.key)) result.warnings = refreshed.get(result.key);
    });
    if (payload.shopping_tryon_batch) applyShoppingTryonBatch(payload.shopping_tryon_batch);
    else renderShoppingTryonPanel();
    toast(`지금 입은 하의를 '${length}'로 반영했어요.`);
  } catch (error) {
    toast(error.message);
  }
}

function shoppingTryonResult(item) {
  const productIds = Array.isArray(item.product_ids) ? item.product_ids : [];
  return {
    ...item,
    key: productIds.join("|"),
    names: productIds.map((productId) =>
      state.shoppingProducts.find((product) => product.product_id === productId)?.name || productId
    ),
  };
}

function stopShoppingTryonBatchPolling() {
  clearInterval(state.shoppingTryonPoll);
  state.shoppingTryonPoll = null;
}

function applyShoppingTryonBatch(batch) {
  state.shoppingTryonBatch = batch;
  state.shoppingTryonResults = (batch.items || [])
    .filter((item) => item.status === "done" && item.image)
    .map(shoppingTryonResult);
  renderShoppingTryonPanel();
  refreshOutfitTryonRenders();
  if (["done", "partial", "failed", "unavailable"].includes(batch.status)) {
    stopShoppingTryonBatchPolling();
  }
}

async function pollShoppingTryonBatch() {
  if (!state.jobId) return stopShoppingTryonBatchPolling();
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}/shopping-tryon-batch`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "무신사 조합 렌더링 상태를 확인하지 못했습니다.");
    applyShoppingTryonBatch(payload);
  } catch (error) {
    stopShoppingTryonBatchPolling();
    toast(error.message);
  }
}

async function startShoppingTryonBatch() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}/shopping-tryon-batch`, {
      method: "POST",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "무신사 전체 조합 렌더링을 시작하지 못했습니다.");
    applyShoppingTryonBatch(payload);
    if (["queued", "running"].includes(payload.status)) {
      stopShoppingTryonBatchPolling();
      state.shoppingTryonPoll = setInterval(pollShoppingTryonBatch, 1200);
    }
  } catch (error) {
    console.warn("무신사 전체 조합 배치를 사용할 수 없습니다:", error);
  }
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

function renderCurrentOutfitEvaluation(evaluation) {
  const matrixBody = $("current-score-matrix");
  const points = $("current-outfit-points");
  if (!evaluation) {
    $("current-score-value").textContent = "—";
    $("current-score-verdict").textContent = "현재 착장 점수를 계산하지 못했습니다.";
    matrixBody.innerHTML = "";
    points.innerHTML = "<li>분석 결과가 충분하지 않아 설명을 만들지 못했습니다.</li>";
    return;
  }

  const score = Number(evaluation.total_score || 0);
  $("current-score-value").textContent = score.toFixed(1);
  $("current-score-verdict").textContent = evaluation.verdict || "Fashion Rules 기반 분석 결과입니다.";
  const labels = {
    body_fit: "체형 적합도",
    situation_fit: "상황 적합도",
    style_fit: "스타일 적합도",
  };
  matrixBody.innerHTML = [
    ["top", "상의"],
    ["bottom", "하의"],
  ].map(([category, categoryLabel]) => {
    const values = evaluation.diagnostic_matrix?.[category] || {};
    const passes = evaluation.pass_matrix?.[category] || {};
    const cells = Object.keys(labels).map((key) => {
      const value = Number(values[key] || 0);
      const passed = passes[key] ?? value >= 85;
      return `<td><span class="matrix-score is-${passed ? "pass" : "fail"}"
        aria-label="${labels[key]} ${value.toFixed(1)}점, ${passed ? "통과" : "보완 필요"}">
        <b>${value.toFixed(1)}</b><small>${passed ? "통과" : "보완"}</small></span></td>`;
    }).join("");
    return `<tr><th scope="row">${categoryLabel}</th>${cells}</tr>`;
  }).join("");

  const summaryPoints = (evaluation.summary_points || []).slice(0, 3);
  points.innerHTML = summaryPoints
    .map((point) => `<li>${escapeHtml(point)}</li>`)
    .join("");
}

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
      <span class="outfit-tag">신발</span>
      <div class="outfit-desc">${escapeHtml(summary["신발"] || "신발 상태를 확인하지 못했습니다.")}</div>
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

function renderFigures(images) {
  const tabs = $("figure-tabs");
  tabs.querySelectorAll(".figure-tab").forEach((tab) => {
    tab.hidden = !images[tab.dataset.image];
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
  };
  $("figure-caption").textContent = (state.result?.mock ? mockCaptions : FIGURE_CAPTIONS)[key] || "";
}

$("delete-now").addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    stopShoppingTryonBatchPolling();
    const response = await fetch(`${API_BASE}/api/jobs/${state.jobId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("삭제하지 못했습니다.");
    const bar = $("privacy-bar");
    bar.classList.add("is-deleted");
    bar.querySelector("strong").textContent = "사진과 결과 이미지를 삭제했습니다.";
    bar.querySelector("span").textContent = "화면에 남은 분석 내용은 새로고침하면 사라집니다.";
    $("delete-now").disabled = true;
    document.querySelectorAll(".figure img, .outfit-tryon-result img")
      .forEach((img) => img.removeAttribute("src"));
    $("figure-caption").textContent = "삭제되었습니다.";
    $("clear-image").click();
    $("clear-body-image").click();
    $("wardrobe-list").replaceChildren();
    state.jobId = null;
    state.shoppingProducts = [];
    state.shoppingOutfits = [];
    state.shoppingTryonResults = [];
    state.shoppingTryonBatch = null;
    $("shopping-tryon-panel").replaceChildren();
    $("shopping-tryon-panel").hidden = true;
    toast("사진을 삭제했습니다.");
  } catch (error) {
    toast(error.message);
  }
});

$("restart").addEventListener("click", () => {
  stopShoppingTryonBatchPolling();
  $("clear-image").click();
  $("clear-body-image").click();
  $("wardrobe-list").replaceChildren();
  state.result = null;
  state.jobId = null;
  state.shoppingProducts = [];
  state.shoppingOutfits = [];
  state.shoppingTryonResults = [];
  state.shoppingTryonBatch = null;
  $("shopping-tryon-panel").replaceChildren();
  $("shopping-tryon-panel").hidden = true;
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

/* 추천 근거는 문장 두 개를 공백으로 이어 붙인 값이다(outfit_combination_recommender.py).
   한 문단으로 흘리면 문장 경계가 아니라 아무 데서나 줄이 넘어가 읽기 나쁘다.
   문장마다 한 줄씩 준다. */
function sentenceLines(text) {
  return String(text ?? "")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)
    .map((sentence) => `<span class="summary-sentence">${escapeHtml(sentence)}</span>`)
    .join("");
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

  $("f-min-budget").addEventListener("input", (event) => syncBudgetSlider(event.currentTarget));
  $("f-max-budget").addEventListener("input", (event) => syncBudgetSlider(event.currentTarget));
  syncBudgetSlider();

  fillSelect("f-purpose", options.purposes);
  fillSelect("f-gender", options.genders);
  fillSelect("f-style", options.styles);
  fillSelect("f-season", options.seasons);
  fillSelect("f-activity", options.activity_levels);
  // Do not pre-select required fields. Insert placeholder for required selects.
  ["f-gender", "f-purpose", "f-style"].forEach((id) => {
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

  // health info suppressed from topbar in this UI
  // fetch(API_BASE + "/api/health").catch(() => {});
})();
