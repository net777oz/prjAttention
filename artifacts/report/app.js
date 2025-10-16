
function $(q, el=document){return el.querySelector(q)}
function $all(q, el=document){return [...el.querySelectorAll(q)]}
function normalize(s){return (s||"").toLowerCase()}

function applyFilters(){
  const q = normalize($("#filter-q").value)
  const min = parseFloat($("#min-score").value||"")
  const key = $("#score-key").value
  const mode = $("#sort-mode").value

  const cards = $all(".card")
  cards.forEach(card=>{
    const name = normalize(card.dataset.name)
    const scoreKey = card.dataset.scoreKey
    const scoreVal = parseFloat(card.dataset.scoreVal || "NaN")
    const hitQ = !q || name.includes(q)
    const hitMin = isNaN(min) || (!isNaN(scoreVal) && scoreVal >= min)
    const hitKey = !key || !scoreKey || scoreKey.endsWith(key) || scoreKey===key
    card.style.display = (hitQ && hitMin && hitKey) ? "" : "none"
  })

  // 정렬
  const grid = $(".grid")
  const shown = $all(".card").filter(c=>c.style.display!=="none")
  shown.sort((a,b)=>{
    const am = parseFloat(a.dataset.mtime), bm = parseFloat(b.dataset.mtime)
    const as = parseFloat(a.dataset.scoreVal || "NaN"), bs = parseFloat(b.dataset.scoreVal || "NaN")
    if(mode==="mtime"){ return (bm - am) }
    if(mode==="score"){
      if(!isNaN(bs) && !isNaN(as)) return (bs - as)
      if(!isNaN(bs)) return -1
      if(!isNaN(as)) return 1
      return (bm - am)
    }
    return 0
  })
  shown.forEach(c=>grid.appendChild(c))

  const any = shown.length>0
  $("#empty").style.display = any ? "none" : ""
}

function init(){
  $all("#filter-q,#min-score,#score-key,#sort-mode").forEach(el=>el.addEventListener("input", applyFilters))
  applyFilters()
}
document.addEventListener("DOMContentLoaded", init)
