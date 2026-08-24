"""
인식 성능 비교용 정답(reference) 라벨링 세트를 만든다.

파이프라인이 검출한 문단마다 원본 이미지에서 잘라낸 crop과, 정답 원문을 적을
빈 입력칸이 나란히 있는 HTML 한 장을 생성한다. 모델이 읽은 결과는 일부러 보여주지
않는다 - 보고 적으면 그 모델 쪽으로 답이 끌려가서 비교가 성립하지 않기 때문이다.

사용법:
    python make_labeling_set.py --json-dir <paragraphs json 폴더> --out <출력 폴더>
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# crop이 너무 작으면 읽기 어려워서 최소 폭까지 확대한다(라벨링 편의용, 평가와 무관)
MIN_DISPLAY_WIDTH = 260


def crop_paragraph(image, bbox, pad):
    w, h = image.size
    x1, y1, x2, y2 = bbox
    return image.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                       min(w, int(x2 + pad)), min(h, int(y2 + pad))))


def to_data_uri(im):
    if im.width < MIN_DISPLAY_WIDTH:
        scale = MIN_DISPLAY_WIDTH / im.width
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    buf = BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_prefill(path):
    """AI 초안 CSV(key, reference_ja, review_note)를 읽어 {key: (초안, 검토메모)}로 돌려준다."""
    import csv
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["key"]: (r.get("reference_ja", ""), r.get("review_note", ""))
                for r in csv.DictReader(f)}


def collect(json_dir, crop_pad):
    items = []
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith("_paragraphs.json"):
            continue
        data = json.load(open(os.path.join(json_dir, name), encoding="utf-8"))
        page = name[: -len("_paragraphs.json")]
        image = Image.open(data["image_path"]).convert("RGB")
        for p in data["paragraphs"]:
            items.append({
                "key": f"{page}#{p['id']}",
                "page": page,
                "id": p["id"],
                "interior": bool(p.get("bubble_bbox")),
                "uri": to_data_uri(crop_paragraph(image, p["bbox"], crop_pad)),
            })
    return items


HTML_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>인식 정답 라벨링</title>
<style>
 body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:0;background:#f6f6f7;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;
        display:flex;gap:16px;align-items:center;z-index:10}
 header b{font-size:15px} #done{color:#0a7}
 button{padding:8px 14px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:14px}
 button.primary{background:#111;color:#fff;border-color:#111}
 main{padding:16px 20px 120px}
 .row{display:flex;gap:16px;align-items:center;background:#fff;border:1px solid #e3e3e3;
      border-radius:8px;padding:12px;margin-bottom:10px}
 .row.filled{border-color:#0a7;background:#f6fffb}
 .row.skipped{opacity:.45}
 .row.needs-review{border-color:#e8a33d;background:#fffdf5}
 .row.edited{border-color:#3d7ae8;background:#f5f9ff}
 .note{font-size:12px;color:#b8730f}
 .idx{width:92px;font-size:12px;color:#666;flex-shrink:0}
 .tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;background:#eee;margin-top:4px}
 img{max-height:220px;border:1px solid #ddd;background:#fff;flex-shrink:0}
 .inp{flex:1;display:flex;flex-direction:column;gap:6px}
 input[type=text]{font-size:18px;padding:10px;border:1px solid #ccc;border-radius:6px;width:100%;
                  font-family:'Yu Gothic','Meiryo',sans-serif}
 label.skip{font-size:13px;color:#666;user-select:none}
 footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ddd;padding:10px 20px}
 textarea{width:100%;height:70px;font-family:Consolas,monospace;font-size:12px}
</style>
<header>
  <b>인식 정답 라벨링</b>
  <span id="done">0 / 0</span>
  <button onclick="save()">진행상황 저장</button>
  <button class="primary" onclick="exportCsv()">CSV 만들기</button>
  <span style="font-size:13px;color:#666">AI가 읽은 초안이 채워져 있습니다. <b>크롭과 대조해서 틀린 것만 고치세요.</b>
    주황색 = 확인 필요. 파란색 = 내가 고친 것.</span>
</header>
<main>
"""

HTML_TAIL = r"""</main>
<footer>
  <textarea id="csv" placeholder="[CSV 만들기]를 누르면 여기에 나옵니다. 전체 복사해서 labels.csv 로 저장하세요."></textarea>
</footer>
<script>
const KEY='it_label_v1';
function rows(){return [...document.querySelectorAll('.row')];}
function count(){
  let n=0; rows().forEach(r=>{
    const v=r.querySelector('input[type=text]').value.trim();
    const sk=r.querySelector('input[type=checkbox]').checked;
    const d=(r.querySelector('input[type=text]').dataset.draft||'').trim();
    r.classList.toggle('filled', !!v && !sk); r.classList.toggle('skipped', sk);
    r.classList.toggle('edited', !!v && v!==d && !sk);
    if(v||sk) n++;
  });
  document.getElementById('done').textContent = n+' / '+rows().length;
}
function save(){
  const d={}; rows().forEach(r=>{d[r.dataset.key]={t:r.querySelector('input[type=text]').value,
                                                   s:r.querySelector('input[type=checkbox]').checked};});
  localStorage.setItem(KEY, JSON.stringify(d)); count();
}
function load(){
  const d=JSON.parse(localStorage.getItem(KEY)||'{}');
  rows().forEach(r=>{const v=d[r.dataset.key]; if(!v)return;
    r.querySelector('input[type=text]').value=v.t||'';
    r.querySelector('input[type=checkbox]').checked=!!v.s;});
  count();
}
function exportCsv(){
  save();
  const esc=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const out=['key,reference_ja,skip'];
  rows().forEach(r=>{
    const t=r.querySelector('input[type=text]').value.trim();
    const s=r.querySelector('input[type=checkbox]').checked?'1':'';
    if(t||s) out.push([esc(r.dataset.key),esc(t),esc(s)].join(','));
  });
  const ta=document.getElementById('csv'); ta.value=out.join('\n'); ta.select();
  try{document.execCommand('copy');}catch(e){}
}
document.addEventListener('input',e=>{if(e.target.matches('input'))count();});
document.addEventListener('change',e=>{if(e.target.matches('input'))save();});
addEventListener('keydown',e=>{if(e.key==='Enter'){const l=[...document.querySelectorAll('input[type=text]')];
  const i=l.indexOf(document.activeElement); if(i>=0&&i<l.length-1){l[i+1].focus();e.preventDefault();}}});
addEventListener('beforeunload',save);
load();
</script>
"""


def esc_attr(v):
    return (v.replace("&", "&amp;").replace('"', "&quot;")
             .replace("<", "&lt;").replace(">", "&gt;"))


def build_html(items, prefill):
    parts = [HTML_HEAD]
    for n, it in enumerate(items, 1):
        tag = "말풍선 안" if it["interior"] else "말풍선 밖"
        draft, note = prefill.get(it["key"], ("", ""))
        cls = "row needs-review" if note else "row"
        note_html = f'<div class="note">확인 필요: {esc_attr(note)}</div>' if note else ""
        parts.append(
            f'<div class="{cls}" data-key="{it["key"]}">'
            f'<div class="idx">{n}. {it["page"]}<br>id={it["id"]}'
            f'<span class="tag">{tag}</span></div>'
            f'<img src="{it["uri"]}">'
            f'<div class="inp"><input type="text" spellcheck="false" placeholder="여기에 일본어 원문"'
            f' value="{esc_attr(draft)}" data-draft="{esc_attr(draft)}">{note_html}'
            f'<label class="skip"><input type="checkbox"> 글자가 아님 / 잘려서 판독 불가 (평가 제외)</label>'
            f'</div></div>\n'
        )
    parts.append(HTML_TAIL)
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="인식 정답 라벨링용 crop + 입력 페이지 생성")
    ap.add_argument("--json-dir", required=True, help="paragraphs json이 있는 폴더")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--prefill", type=str, default=None,
                    help="AI 초안 CSV 경로(key, reference_ja, review_note). 주면 입력칸을 채워둔다.")
    ap.add_argument("--crop-pad", type=int, default=config.GEMINI_CROP_PAD,
                    help=f"crop 여유 픽셀 (기본값: {config.GEMINI_CROP_PAD}, Gemini에 주는 것과 동일)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = collect(args.json_dir, args.crop_pad)

    html_path = os.path.join(args.out, "labeling.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(items, load_prefill(args.prefill)))

    index_path = os.path.join(args.out, "items.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in it.items() if k != "uri"} for it in items],
                  f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(html_path) / 1024 / 1024
    print(f"문단 {len(items)}개")
    print(f"라벨링 페이지: {html_path}  ({size_mb:.1f}MB)")
    print(f"항목 목록    : {index_path}")


if __name__ == "__main__":
    main()
