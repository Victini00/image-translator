"""
번역 품질 블라인드 평가 페이지를 만든다.

사용법:
    python make_mqm_set.py --translations <translations.json> --out <폴더> [--n 40]
"""

import os
import sys
import json
import random
import itertools
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402,F401

TAGS = [
    ("mistrans", "오역", "원문의 의미가 틀림"),
    ("omission", "누락", "원문에 있는 내용이 빠짐"),
    ("tone", "어조", "의미는 맞지만 만화 대사답지 않음(문어체 등)"),
    ("awkward", "어색", "한국어 문장이 부자연스러움"),
]

HEAD = """<!doctype html>
<meta charset="utf-8">
<title>번역 품질 블라인드 평가</title>
<style>
 body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:0;background:#f6f6f7;color:#111}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;
        display:flex;gap:14px;align-items:center;z-index:10;flex-wrap:wrap}
 header b{font-size:15px} #done{color:#0a7;font-weight:600}
 button{padding:8px 14px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:14px}
 button.primary{background:#111;color:#fff;border-color:#111}
 main{padding:16px 20px 130px}
 .item{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:14px 16px;margin-bottom:14px}
 .item.done{border-color:#0a7;background:#f7fffc}
 .src{font-family:'Yu Gothic','Meiryo',sans-serif;font-size:19px;margin-bottom:4px}
 .no{font-size:12px;color:#888;margin-bottom:8px}
 .cand{display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-top:1px dashed #e5e5e5}
 .slot{font-weight:700;color:#555;width:18px;flex-shrink:0;padding-top:3px}
 .txt{flex:1;font-size:16px;line-height:1.5}
 .ctl{display:flex;gap:14px;align-items:center;flex-shrink:0;flex-wrap:wrap;justify-content:flex-end}
 .stars label{cursor:pointer;padding:3px 7px;border:1px solid #ccc;border-radius:5px;font-size:13px;margin-left:-1px}
 .stars input{display:none}
 .stars input:checked + span{background:#111;color:#fff;border-radius:4px;padding:2px 6px}
 .tags label{font-size:12px;color:#666;margin-left:8px;cursor:pointer;white-space:nowrap}
 footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ddd;padding:10px 20px}
 textarea{width:100%;height:74px;font-family:Consolas,monospace;font-size:12px}
</style>
<header>
  <b>번역 품질 블라인드 평가</b>
  <span id="done">0 / 0</span>
  <button onclick="save()">진행상황 저장</button>
  <button class="primary" onclick="exportCsv()">CSV 만들기</button>
  <span style="font-size:13px;color:#666">
    <b>5</b>=그대로 써도 됨 · <b>4</b>=사소한 아쉬움 · <b>3</b>=뜻은 통함 ·
    <b>2</b>=고쳐야 함 · <b>1</b>=못 씀. 3점 이하면 이유 태그도 눌러주세요.</span>
</header>
<main>
"""

TAIL = r"""</main>
<footer>
  <textarea id="csv" placeholder="[CSV 만들기]를 누르면 여기에 나옵니다. 전체 복사해서 mqm.csv 로 저장하세요."></textarea>
</footer>
<script>
const KEY='it_mqm_v1';
function items(){return [...document.querySelectorAll('.item')];}
function count(){
  let n=0;
  items().forEach(it=>{
    const cs=[...it.querySelectorAll('.cand')];
    const ok=cs.every(c=>c.querySelector('input[type=radio]:checked'));
    it.classList.toggle('done', ok); if(ok) n++;
  });
  document.getElementById('done').textContent=n+' / '+items().length;
}
function state(){
  const d={};
  document.querySelectorAll('.cand').forEach(c=>{
    const r=c.querySelector('input[type=radio]:checked');
    const tags=[...c.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value);
    if(r||tags.length) d[c.dataset.cid]={s:r?r.value:'',t:tags};
  });
  return d;
}
function save(){localStorage.setItem(KEY,JSON.stringify(state()));count();}
function load(){
  const d=JSON.parse(localStorage.getItem(KEY)||'{}');
  document.querySelectorAll('.cand').forEach(c=>{
    const v=d[c.dataset.cid]; if(!v)return;
    if(v.s){const r=c.querySelector('input[type=radio][value="'+v.s+'"]'); if(r)r.checked=true;}
    (v.t||[]).forEach(t=>{const b=c.querySelector('input[type=checkbox][value="'+t+'"]'); if(b)b.checked=true;});
  });
  count();
}
function exportCsv(){
  save();
  const esc=s=>'"'+String(s).replace(/"/g,'""')+'"';
  const out=['key,system,score,tags'];
  document.querySelectorAll('.cand').forEach(c=>{
    const r=c.querySelector('input[type=radio]:checked'); if(!r)return;
    const tags=[...c.querySelectorAll('input[type=checkbox]:checked')].map(x=>x.value).join('|');
    out.push([esc(c.dataset.key),esc(c.dataset.sys),r.value,esc(tags)].join(','));
  });
  const ta=document.getElementById('csv'); ta.value=out.join('\n'); ta.select();
  try{document.execCommand('copy');}catch(e){}
}
document.addEventListener('change',e=>{if(e.target.matches('input'))save();});
addEventListener('beforeunload',save);
load();
</script>
"""


def esc(v):
    return (str(v).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def build(data, n, seed):
    sources = data["sources"]
    systems = list(data["systems"])
    keys = [k for k in sources if all(k in data["systems"][s] for s in systems)]

    rng = random.Random(seed)
    rng.shuffle(keys)
    keys = keys[:n]

    # 시스템 순서를 항목마다 바꾼다. 특정 슬롯(A)에 늘 같은 시스템이 오면 평가자가
    # 패턴을 학습해서 블라인드가 깨진다. 그냥 매번 섞으면 우연히 한쪽으로 쏠리므로
    # (실측: C 슬롯에 한 시스템이 절반), 가능한 순열을 고르게 돌려서 슬롯별 등장
    # 횟수를 맞춘다.
    perms = list(itertools.permutations(systems))
    schedule = [perms[i % len(perms)] for i in range(len(keys))]
    rng.shuffle(schedule)

    parts = [HEAD]
    for i, key in enumerate(keys, 1):
        order = list(schedule[i - 1])

        parts.append('<div class="item">')
        parts.append(f'<div class="no">{i} / {len(keys)} &nbsp; {esc(key)}</div>')
        parts.append(f'<div class="src">{esc(sources[key])}</div>')

        for slot, sysname in zip("ABC", order):
            cid = f"{key}|{sysname}"
            stars = "".join(
                f'<label><input type="radio" name="{esc(cid)}" value="{v}">'
                f'<span>{v}</span></label>' for v in range(1, 6)
            )
            tags = "".join(
                f'<label title="{esc(desc)}"><input type="checkbox" value="{tid}"> {label}</label>'
                for tid, label, desc in TAGS
            )
            parts.append(
                f'<div class="cand" data-cid="{esc(cid)}" data-key="{esc(key)}" data-sys="{esc(sysname)}">'
                f'<div class="slot">{slot}</div>'
                f'<div class="txt">{esc(data["systems"][sysname][key])}</div>'
                f'<div class="ctl"><span class="stars">{stars}</span>'
                f'<span class="tags">{tags}</span></div></div>'
            )
        parts.append("</div>\n")

    parts.append(TAIL)
    return "".join(parts), keys


def main():
    ap = argparse.ArgumentParser(description="번역 품질 블라인드 평가 페이지 생성")
    ap.add_argument("--translations", required=True, help="run_translations.py 결과 json")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--n", type=int, default=40, help="평가할 항목 수 (기본 40)")
    ap.add_argument("--seed", type=int, default=20260907,
                    help="표본 추출·순서 섞기 시드. 같은 값을 주면 같은 페이지가 나온다.")
    args = ap.parse_args()

    data = json.load(open(args.translations, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)

    html, keys = build(data, args.n, args.seed)
    path = os.path.join(args.out, "mqm.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"시스템 {len(data['systems'])}개: {', '.join(data['systems'])}")
    print(f"평가 항목 {len(keys)}개 (전체 {len(data['sources'])}개 중, seed={args.seed})")
    print(f"판정 횟수 {len(keys) * len(data['systems'])}회")
    print(f"페이지: {path}")


if __name__ == "__main__":
    main()
