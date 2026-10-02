"use strict";

const el = (id) => document.getElementById(id);
let race = null;
let currentLap = 1;

function fmtTime(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function pct(x) { return `${Math.round(x * 100)}%`; }

/** Running order at the selected lap, or the latest lap at or before it. */
function orderAt(lap) {
  let best = null;
  for (const row of race.laps) if (row.lap <= lap && (!best || row.lap > best.lap)) best = row;
  return best;
}

function renderOrder(lap) {
  const row = orderAt(lap);
  const body = el("orderTable").querySelector("tbody");
  body.innerHTML = "";
  el("orderNote").textContent = row ? `end of lap ${row.lap}` : "";
  if (!row) return;
  for (const car of row.order) {
    const tr = document.createElement("tr");
    const age = car.age === null ? "" : `${car.age}`;
    tr.innerHTML =
      `<td class="pos">${car.pos}</td>` +
      `<td class="drv">${car.driver}</td>` +
      `<td><span class="tyre ${car.compound}">${car.compound[0]}${age}</span>` +
      `${car.pit ? '<span class="pitting">in pit</span>' : ""}</td>` +
      `<td class="gap">${car.pos === 1 ? "leader" : "+" + car.gap_s.toFixed(1)}</td>`;
    body.appendChild(tr);
  }
}

/** P(pit better) across the three cross-plan correlations, as a range, never a verdict. */
function probBar(p) {
  const vals = Object.values(p);
  const lo = Math.min(...vals), hi = Math.max(...vals), mid = p["0.5"];
  const span = `<span class="span" style="left:${lo * 100}%;width:${(hi - lo) * 100}%"></span>`;
  const half = `<span class="half"></span>`;
  const marker = `<span class="mid" style="left:${mid * 100}%"></span>`;
  return (
    `<div class="prange">P(pit better) <b>${pct(lo)} to ${pct(hi)}</b></div>` +
    `<div class="bar">${span}${half}${marker}</div>` +
    `<div class="kv">across error correlation 0 / 0.5 / 0.9: ` +
    `<span>${pct(p["0.0"])}, ${pct(p["0.5"])}, ${pct(p["0.9"])}</span></div>`
  );
}

function verdictClass(text) {
  if (text.startsWith("overlapping")) return "overlap";
  if (text.startsWith("no ") || text.startsWith("race")) return "blind";
  return text === "pit" ? "pit" : "stay";
}

function verdictText(card) {
  if (card.decision.startsWith("no anchor")) {
    return "no decision: model has no anchor yet (laps 2-4 skipped)";
  }
  if (card.decision.startsWith("no data")) return "no decision: no laps yet";
  if (card.decision.startsWith("race")) return "no decision: race ending";
  return card.decision;
}

function renderCard(card) {
  const cls = verdictClass(card.decision);
  let html =
    `<div class="drv">${card.driver}` +
    (card.compound_now ? ` <span class="tyre ${card.compound_now}">` +
      `${card.compound_now[0]}${card.tyre_age !== null ? card.tyre_age : ""}</span>` : "") +
    `</div><div class="verdict ${cls}">${verdictText(card)}</div>`;
  if (card.decided) {
    html += probBar(card.p);
    html += `<div class="kv">median gain if the call is real: ` +
      `<span>${card.gain_if_real > 0 ? "+" : ""}${card.gain_if_real.toFixed(1)} s</span>, ` +
      `if false: <span>${card.gain_if_false > 0 ? "+" : ""}` +
      `${card.gain_if_false.toFixed(1)} s</span></div>`;
    html += `<div class="kv">pit now: <span>${card.pit_now_plan}</span><br>` +
      `stay out: <span>${card.stay_out_plan}</span></div>`;
    if (card.p_soft_bias) {
      html += `<div class="kv">with soft p50 +0.075 s: ` +
        `<span>${pct(card.p_soft_bias["0.5"])}</span> at correlation 0.5</div>`;
    }
    if (card.share_h_gt_30 > 0) {
      html += `<div class="kv">${pct(card.share_h_gt_30)} of plan laps are more than 30 ` +
        `laps ahead, where the tyre intervals are a floor: this P is overconfident</div>`;
    }
  }
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = html;
  return div;
}

function renderEvents(lap) {
  const box = el("events");
  box.innerHTML = "";
  const shown = race.events.filter((e) => e.lap <= lap);
  if (!shown.length) {
    box.innerHTML = `<p class="note">No race control activity yet. Fast Flag has made no ` +
      `call and race control has deployed nothing up to lap ${lap}.</p>`;
    return;
  }
  for (const e of shown.slice().reverse()) {
    const div = document.createElement("div");
    div.className = `event ${e.kind}`;
    let meta = `lap ${e.lap}, ${fmtTime(e.t)} session time. ${e.reason}`;
    if (e.kind === "call" && e.p_call_real !== null && e.p_call_real !== undefined) {
      meta += `<br>Carried as a probability: P(a neutralisation follows) = ` +
        `${pct(e.p_call_real)}` +
        (e.p_call_real_range && e.p_call_real_range[0] !== null
          ? ` (90% ${pct(e.p_call_real_range[0])} to ${pct(e.p_call_real_range[1])})`
          : "") + `, Fast Flag's own out-of-sample precision.`;
    }
    if (e.kind === "call" && e.outcome_lap !== undefined && lap >= e.outcome_lap) {
      meta += e.followed_by
        ? `<br><b>Outcome:</b> race control deployed a ${e.followed_by} after this call.`
        : `<br><b>Outcome:</b> no neutralisation followed. This was a false call, and the ` +
          `cost of acting on it is the cost of a green-flag stop.`;
    }
    if (e.kind === "deploy") {
      meta += `<br>Official neutralisation` +
        (e.lap_end ? `, ran to lap ${e.lap_end}.` : ".") +
        ` Decisions here are the no-call path: what the system does when the feed is ` +
        `missed or ignored.`;
    }
    div.innerHTML = `<h4>${e.label}</h4><div class="meta">${meta}</div>`;
    const cards = document.createElement("div");
    cards.className = "cards";
    if (e.decisions.length) {
      for (const c of e.decisions) cards.appendChild(renderCard(c));
    } else {
      cards.innerHTML = `<p class="note">No decisions recorded at this event.</p>`;
    }
    div.appendChild(cards);
    box.appendChild(div);
  }
}

function setLap(lap) {
  currentLap = Math.max(1, Math.min(race.race_laps, lap));
  el("lap").value = currentLap;
  el("lapLabel").textContent = `lap ${currentLap} of ${race.race_laps}`;
  renderOrder(currentLap);
  renderEvents(currentLap);
}

async function loadRace(rid) {
  race = await (await fetch(`/api/race/${rid}`)).json();
  el("lap").max = race.race_laps;
  const neutral = race.neutral.map((n) =>
    `${n.kind} lap ${n.lap_deploy}${n.lap_end ? " to " + n.lap_end : ""}`).join(", ");
  el("scope").textContent =
    `${race.season} ${race.event}: ${race.race_laps} laps, median clean lap ` +
    `${race.median_lap_s} s. Official neutralisations: ${neutral || "none"}. ` +
    `Holdout race: nothing here was fitted on it.`;
  el("accounts").innerHTML = race.accounts_for.map((x) => `<li>${x}</li>`).join("");
  el("notAccounts").innerHTML = race.not_accounted.map((x) => `<li>${x}</li>`).join("");
  const first = race.events.length ? race.events[0] : null;
  el("jump").style.display = first ? "" : "none";
  if (first) el("jump").textContent = `jump to the ${first.kind === "call" ? "call" : "event"}`;
  setLap(first ? Math.max(1, first.lap - 1) : 1);
}

async function init() {
  const index = await (await fetch("/api/races")).json();
  const sel = el("race");
  sel.innerHTML = index.races
    .map((r) => `<option value="${r.race}">${r.season} ${r.event}</option>`).join("");
  sel.onchange = () => loadRace(sel.value);
  el("lap").oninput = (e) => setLap(Number(e.target.value));
  el("prev").onclick = () => setLap(currentLap - 1);
  el("next").onclick = () => setLap(currentLap + 1);
  el("jump").onclick = () => { if (race.events.length) setLap(race.events[0].lap); };
  document.onkeydown = (e) => {
    if (e.key === "ArrowLeft") setLap(currentLap - 1);
    if (e.key === "ArrowRight") setLap(currentLap + 1);
  };
  await loadRace(index.races[0].race);
}

init();
