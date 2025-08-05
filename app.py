# app.py
import os
import uuid
from datetime import timedelta
from urllib.parse import urlencode, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, session,
    jsonify, redirect, url_for
)

# ----------------------------
# OpenAI (v1/new API 우선, 실패 시 구 API fallback)
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

_client = None
_use_legacy = False
try:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    # old SDK fallback
    import openai
    openai.api_key = OPENAI_API_KEY
    _use_legacy = True

def call_openai_chat(messages, model="gpt-4o-nano", temperature=0.6):
    """messages = [{'role':'system|user|assistant','content':'...'}]"""
    try:
        if not OPENAI_API_KEY:
            return "(오류) OPENAI_API_KEY가 설정되지 않았습니다."
        if _use_legacy:
            import openai
            r = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return r["choices"][0]["message"]["content"].strip()
        else:
            r = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(모델 오류) {e}"

# ----------------------------
# Flask
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.permanent_session_lifetime = timedelta(days=7)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")

# ----------------------------
# 네이버 크롤링
# ----------------------------
def crawl_naver_place(query: str):
    """네이버 통합검색에서 플레이스 정보 스크래핑 (정적 DOM 기준)"""
    search_url = f"https://search.naver.com/search.naver?{urlencode({'query': query})}"
    headers = {"User-Agent": UA}
    res = requests.get(search_url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    def text(sel):
        el = soup.select_one(sel)
        return el.get_text(" ", strip=True) if el else None

    # 메뉴는 덩어리가 커서 요약해서 일부만 사용
    menu_block = soup.select_one(".place_section_content .ds3HZ")
    menu_text = menu_block.get_text(" / ", strip=True) if menu_block else None
    if menu_text and len(menu_text) > 400:
        menu_text = menu_text[:400] + "…"

    # 비슷한 맛집들
    similar_places = [n.get_text(" ", strip=True)
                      for n in soup.select(".KFSbV")
                      if n.get_text(strip=True)]

    # 제목 보강
    title = soup.select_one(".place_section_title span")
    name = title.get_text(strip=True) if title else query

    result = {
        "name": name,
        "address": text(".LDgIH") or "정보 없음",
        "hours": text(".A_cdD em") or "정보 없음",
        "tel": text(".xlx7Q") or "정보 없음",
        "rating": text(".score") or "정보 없음",
        "menu": menu_text or "정보 없음",
        "similar_places": similar_places,
        "link": search_url
    }
    return result

def crawl_blog_reviews(query: str, max_count: int = 3):
    """네이버 블로그 검색 → 상위 글 본문 일부 텍스트 추출(모바일 뷰 우선)"""
    headers = {"User-Agent": UA}
    search_url = f"https://search.naver.com/search.naver?{urlencode({'query': f'{query} 블로그'})}"
    r = requests.get(search_url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = [a["href"] for a in soup.select("a.total_tit")
             if a.has_attr("href") and "blog.naver.com" in a["href"]]

    texts = []
    for link in links[:max_count]:
        try:
            if "blog.naver.com" in link and not link.startswith("https://m.blog.naver.com"):
                p = urlparse(link)
                mobile = f"https://m.{p.netloc}{p.path}"
                link = mobile if not p.query else f"{mobile}?{p.query}"

            br = requests.get(link, headers=headers, timeout=10, allow_redirects=True)
            bs = BeautifulSoup(br.text, "html.parser")

            node = bs.select_one("span.sds-comps-text-content")  # 주신 선택자
            content = node.get_text(" ", strip=True) if node else bs.get_text(" ", strip=True)
            if content:
                texts.append(content[:3000])
        except Exception as e:
            print("blog fetch failed:", e)
            continue
    return texts

def summarize_texts(texts):
    if not texts:
        return "요약할 블로그를 찾지 못했습니다."
    joined = "\n\n---\n\n".join(texts)
    prompt = (
        "다음은 특정 장소에 대한 블로그 후기입니다. "
        "메뉴/맛/서비스/분위기/가격/대기/주차 중심으로 장점·단점을 5줄 이내 bullet로 한국어로 요약해줘.\n\n"
        f"{joined}"
    )
    return call_openai_chat(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        temperature=0.3
    )

# ----------------------------
# 대화방(세션 메모리 + 요약 압축)
# ----------------------------
def _ensure_store():
    session.permanent = True
    if "convs" not in session:
        cid = str(uuid.uuid4())[:8]
        session["convs"] = [{
            "id": cid,
            "title": "New Chat",
            "messages": [
                {"role": "system", "content": "당신은 친절한 로컬 가이드입니다. 한국어로 간결히 답하세요."}
            ]
        }]
        session["current"] = cid

def _get_conv(cid: str):
    for c in session.get("convs", []):
        if c["id"] == cid:
            return c
    return None

def chat_with_llm(messages):
    return call_openai_chat(messages, model="gpt-4o-mini", temperature=0.6)

def compress_history_if_needed_for(conv):
    hist = conv["messages"]
    MAX_TURNS = 16
    if len(hist) <= MAX_TURNS:
        return
    head = hist[:-8]
    tail = hist[-8:]
    summary_prompt = [
        {"role": "system", "content": "너는 대화 요약가다. 사용자 선호/사실/미완 요청을 간결 메모로 정리해라."},
        {"role": "user", "content": "\n\n".join([f"{m['role']}: {m['content']}" for m in head])}
    ]
    memo = chat_with_llm(summary_prompt)
    session["chat_memory_note"] = memo
    conv["messages"] = [{"role": "system", "content": f"(요약 메모) {memo}"}] + tail

# ----------------------------
# 라우팅: 검색형 페이지
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = (request.form.get("query") or "").strip()
        if not keyword:
            return render_template("index.html", error="검색어를 입력해 주세요.")
        place = crawl_naver_place(keyword)
        reviews = crawl_blog_reviews(keyword, max_count=3)
        summary = summarize_texts(reviews)
        return render_template("result.html", place=place, summary=summary)
    return render_template("index.html")

# ----------------------------
# 라우팅: 대화형 UI
# ----------------------------
@app.route("/chat")
def chat_root():
    _ensure_store()
    return redirect(url_for("chat_room", cid=session["current"]))

@app.route("/chat/<cid>", methods=["GET"])
def chat_room(cid):
    _ensure_store()
    conv = _get_conv(cid)
    if not conv:
        return redirect(url_for("chat_root"))
    session["current"] = cid
    return render_template("chat.html", convs=session["convs"], active=conv)

@app.route("/chat/new", methods=["POST"])
def chat_new():
    _ensure_store()
    cid = str(uuid.uuid4())[:8]
    new_conv = {
        "id": cid,
        "title": request.form.get("title") or "New Chat",
        "messages": [
            {"role": "system", "content": "당신은 친절한 로컬 가이드입니다. 한국어로 간결히 답하세요."}
        ]
    }
    session["convs"].insert(0, new_conv)
    session["current"] = cid
    session.modified = True
    return jsonify({"cid": cid})

@app.route("/chat/<cid>/rename", methods=["POST"])
def chat_rename(cid):
    _ensure_store()
    title = (request.form.get("title") or "").strip()[:60]
    conv = _get_conv(cid)
    if conv and title:
        conv["title"] = title
        session.modified = True
    return jsonify({"ok": True, "title": conv["title"] if conv else ""})

@app.route("/chat/<cid>/send", methods=["POST"])
def chat_send_to_room(cid):
    _ensure_store()
    conv = _get_conv(cid)
    if not conv:
        return jsonify({"error": "대화방을 찾을 수 없습니다."}), 404

    user_text = (request.form.get("message") or "").strip()
    if not user_text:
        return jsonify({"error": "메시지를 입력해 주세요."}), 400

    # 사용자 메시지 추가
    conv["messages"].append({"role": "user", "content": user_text})

    # 요약 메모 포함
    mem = session.get("chat_memory_note")
    messages = []
    if mem:
        messages.append({"role": "system", "content": f"(요약 메모) {mem}"})
    messages += conv["messages"]

    # LLM 응답
    answer = chat_with_llm(messages)

    # 응답 추가
    conv["messages"].append({"role": "assistant", "content": answer})
    session.modified = True

    # 히스토리 길이 관리
    compress_history_if_needed_for(conv)

    # 첫 유저 발화 시 제목 자동 생성
    if conv["title"] == "New Chat" and len([m for m in conv["messages"] if m["role"] == "user"]) == 1:
        conv["title"] = user_text[:30]

    return jsonify({"reply": answer})

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    # 템플릿: index.html / result.html / chat.html 필요
    # 실행: python app.py  →  /  혹은  /chat
    app.run(debug=False)
