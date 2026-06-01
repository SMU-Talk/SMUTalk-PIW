const authModal = document.querySelector("#authModal");
const guestNote = document.querySelector("#guestNote");
const statusPill = document.querySelector("#statusPill");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const toast = document.querySelector("#toast");
const sendButton = document.querySelector(".send-button");
const openAuthButtons = [...document.querySelectorAll("[data-open-auth]")];

let sessionReady = false;
let currentMode = "guest";

const MENU_GUIDE = {
  "상명소개": {
    prompt: "상명소개에서 어떤 내용을 알고 싶으신가요?",
    groups: [
      ["열린 총장실", "총장인사말/프로필", "총장동정"],
      ["발전계획", "상명 2027"],
      ["역사ㆍ상징", "연혁", "교육이념", "인재상", "상명상징", "UI디자인", "교가 및 학원가"],
      ["세계 속 상명"],
      ["대학현황", "일반현황 및 주요지표", "기구표", "대학자체평가", "학교규정", "상명요람", "학교법인"],
      ["캠퍼스 안내", "상명 갤러리", "찾아오시는 길", "캠퍼스투어 신청", "교내 전화번호 검색", "시설 대관 안내"],
      ["정보공개", "정보공개제도 안내", "사전공개", "정보공개청구"],
      ["개인정보처리방침", "개인정보처리방침", "개인정보위탁현황", "개인정보제3자제공현황"],
      ["입찰ㆍ채용", "입찰", "채용"],
    ],
  },
  "입학안내": {
    prompt: "입학안내에서 어떤 입학 정보를 찾으시나요?",
    groups: [
      ["WHY 상명"],
      ["대학 입학"],
      ["대학원 입학"],
      ["International Student", "학부 입학", "대학원 입학", "한국어 연수"],
    ],
  },
  "대학ㆍ대학원": {
    prompt: "대학ㆍ대학원에서 어느 영역을 확인할까요?",
    groups: [
      ["대학소개", "서울캠퍼스", "천안캠퍼스"],
      ["대학원소개"],
      ["평생교육", "미래교육원(서울)", "미래교육원(천안)"],
    ],
  },
  "연구ㆍ산학": {
    prompt: "연구ㆍ산학에서 어떤 정보를 안내할까요?",
    groups: [
      ["산학협력", "산학협력단", "우수 사업", "우수 연구자"],
      ["교수프로필"],
      ["부설연구기관", "서울캠퍼스", "천안캠퍼스"],
    ],
  },
  "학사안내": {
    prompt: "학사안내에서 어떤 학사 항목이 궁금하신가요?",
    groups: [
      ["학사일정"],
      ["학사", "학사제도 및 강의시간표", "교육과정", "전공제도", "수업 및 수강신청", "학적변동", "성적", "졸업", "교직", "등록안내"],
      ["장학", "장학금지급규정", "교내 장학금", "교외 장학금", "학자금 융자제도"],
      ["행정서식"],
    ],
  },
  "대학생활": {
    prompt: "대학생활에서 어떤 서비스를 찾으시나요?",
    groups: [
      ["통합공지"],
      ["상명 Q&A", "자주하는 질문", "질문/답변"],
      ["대학행사", "신입생 오리엔테이션"],
      ["학생지원", "병무행정", "식당메뉴", "버스안내", "학생증발급", "국제학생증"],
      ["학생활동", "학생자치기구", "동아리정보", "교내생활", "사회봉사"],
      ["IT 서비스", "전자출결시스템", "SM-EDU 서비스", "모바일 서비스", "통합보안프로그램", "클라우드메일", "IT서비스 이용안내", "홈페이지 개선요청"],
      ["글로벌 프로그램", "글로벌 프로그램", "글로벌 커뮤니티", "국제교류 프로그램"],
      ["진로취업 서비스", "진로취업 솔루션", "채용정보"],
      ["학생생활관", "학생생활관(서울)", "학생생활관(천안)"],
    ],
  },
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    credentials: "same-origin",
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "요청을 처리하지 못했습니다.");
  }
  return data;
}

function setSessionState(me) {
  sessionReady = Boolean(me.authenticated);
  currentMode = me.mode || "guest";
  messageInput.disabled = !sessionReady;
  sendButton.disabled = !sessionReady;
  messageInput.placeholder = sessionReady ? "상명대학교에 대해 질문해 보세요" : "게스트 모드를 준비하고 있습니다";

  const isGuest = currentMode === "guest";
  guestNote.textContent = isGuest ? "게스트 모드로 이용 중" : `${me.name || "사용자"}님 로그인 중`;
  statusPill.textContent = isGuest ? "게스트 모드" : "로그인 모드";
  openAuthButtons.forEach((button) => {
    button.textContent = isGuest ? "로그인" : "로그아웃";
  });
}

function openAuthModal() {
  authModal.classList.remove("is-hidden");
  authModal.setAttribute("aria-hidden", "false");
  const firstInput = authModal.querySelector("input");
  window.setTimeout(() => firstInput?.focus(), 80);
}

function closeAuthModal() {
  authModal.classList.add("is-hidden");
  authModal.setAttribute("aria-hidden", "true");
}

function buildGuideText(category) {
  const guide = MENU_GUIDE[category];
  if (!guide) return "";
  return `${guide.prompt}\n아래 세부 항목을 선택하거나, 원하는 내용을 직접 입력해 주세요.`;
}

function buildGuideOptions(category) {
  const guide = MENU_GUIDE[category];
  if (!guide) return [];
  return guide.groups.flatMap(([group, ...items]) => {
    if (items.length === 0) return [{ label: group, query: `${category} ${group} 알려줘` }];
    return items.map((item) => ({ label: item, query: `${category} ${group} ${item} 알려줘` }));
  });
}

function addMessage(role, content, options = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";

  if (role === "assistant") {
    const image = document.createElement("img");
    image.src = "/assets/SMU.png";
    image.alt = "";
    avatar.append(image);
  } else {
    avatar.textContent = "나";
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const text = document.createElement("div");
  text.textContent = content;
  bubble.append(text);

  if (options.length > 0) {
    const optionList = document.createElement("div");
    optionList.className = "menu-options";
    options.forEach((option) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "menu-option";
      item.textContent = option.label;
      item.addEventListener("click", () => {
        messageInput.value = option.query;
        messageInput.focus();
      });
      optionList.append(item);
    });
    bubble.append(optionList);
  }

  article.append(avatar, bubble);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
}

function resetMessages() {
  messages.innerHTML = "";
  addMessage(
    "assistant",
    "안녕하세요. SMU Talk입니다. 현재 게스트 모드로 이용 중입니다. 입학, 학사, 장학금, 캠퍼스, 도서관, 포털 관련 질문을 입력해 주세요.",
  );
}

async function refreshHistory() {
  const data = await api("/api/history");
  resetMessages();
  for (const message of data.messages) {
    addMessage(message.role, message.content);
  }
}

async function startGuestSession() {
  const data = await api("/api/guest", {
    method: "POST",
    body: JSON.stringify({ guestName: "게스트" }),
  });
  setSessionState({ authenticated: true, mode: data.mode, name: data.name });
  resetMessages();
}

async function refreshMe() {
  const me = await api("/api/me");
  if (!me.authenticated) {
    await startGuestSession();
    return;
  }
  setSessionState(me);
  await refreshHistory();
}

async function logoutToGuest() {
  await api("/api/logout", { method: "POST", body: "{}" });
  await startGuestSession();
  showToast("로그아웃되었습니다.");
}

document.querySelectorAll("[data-auth-tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll("[data-auth-tab]").forEach((item) => item.classList.remove("is-active"));
    document.querySelectorAll(".auth-form").forEach((form) => form.classList.remove("is-active"));
    tab.classList.add("is-active");
    document.querySelector(`#${tab.dataset.authTab}Form`).classList.add("is-active");
    document.querySelector("#authTitle").textContent = tab.dataset.authTab === "login" ? "로그인" : "회원가입";
  });
});

openAuthButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    if (currentMode === "guest") {
      openAuthModal();
      return;
    }

    try {
      await logoutToGuest();
    } catch (error) {
      showToast(error.message);
    }
  });
});
document.querySelectorAll("[data-close-auth]").forEach((button) => button.addEventListener("click", closeAuthModal));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !authModal.classList.contains("is-hidden")) {
    closeAuthModal();
  }
});

document.querySelector("#guestModeButton").addEventListener("click", async () => {
  try {
    if (currentMode !== "guest") {
      await logoutToGuest();
    }
    closeAuthModal();
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  try {
    const data = await api("/api/login", {
      method: "POST",
      body: JSON.stringify(Object.fromEntries(formData)),
    });
    setSessionState({ authenticated: true, mode: data.mode, name: data.name });
    closeAuthModal();
    await refreshHistory();
    showToast("로그인되었습니다.");
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  try {
    const data = await api("/api/register", {
      method: "POST",
      body: JSON.stringify(Object.fromEntries(formData)),
    });
    setSessionState({ authenticated: true, mode: data.mode, name: data.name });
    closeAuthModal();
    await refreshHistory();
    showToast("계정이 생성되었습니다.");
  } catch (error) {
    showToast(error.message);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || !sessionReady) {
    return;
  }

  addMessage("user", message);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendButton.disabled = true;

  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    addMessage("assistant", data.reply);
  } catch (error) {
    showToast(error.message);
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 160)}px`;
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelector("#clearButton").addEventListener("click", async () => {
  await api("/api/clear", { method: "POST", body: "{}" });
  resetMessages();
  showToast("대화를 지웠습니다.");
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    const category = button.textContent.trim();
    if (MENU_GUIDE[category]) {
      addMessage("assistant", buildGuideText(category), buildGuideOptions(category));
      messages.scrollTop = messages.scrollHeight;
      return;
    }
    messageInput.value = button.dataset.prompt;
    messageInput.focus();
  });
});

refreshMe().catch(() => {
  setSessionState({ authenticated: false });
  showToast("게스트 모드를 시작하지 못했습니다. 서버 상태를 확인해 주세요.");
});
