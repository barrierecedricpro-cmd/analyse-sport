#!/usr/bin/env python3
"""Analyse sportive automatique : foot (5 grands championnats) et NHL.

Sources gratuites, sans clé :
  - API publique d'ESPN : matchs, résultats, classements, buteurs et passeurs, cotes (foot et NHL) ;
  - API officielle de la NHL : calendrier, classement, stats des joueurs, composition du dernier match.

Chaque exécution :
  1. complète les résultats des jours passés dans docs/history.json ;
  2. récupère les matchs du jour et leurs cotes, puis convertit les cotes en probabilités (marge retirée) ;
  3. calcule buts attendus, scores probables, buteurs et joueurs décisifs ;
  4. calcule le bilan : les probabilités annoncées se vérifient-elles ?
  5. écrit docs/data.json, que le site docs/index.html affiche.

Usage : python app.py
"""
import datetime as dt
import difflib
import json
import math
import os
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

import requests
from scipy.optimize import brentq, fsolve
from scipy.stats import poisson

PARIS = ZoneInfo("Europe/Paris")
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
DATA_FILE = os.path.join(DOCS, "data.json")
HIST_FILE = os.path.join(DOCS, "history.json")

# (nom affiché, code de championnat ESPN)
LEAGUES = [
    ("Ligue 1", "fra.1"),
    ("Premier League", "eng.1"),
    ("LaLiga", "esp.1"),
    ("Serie A", "ita.1"),
    ("Bundesliga", "ger.1"),
    ("Ligue des champions", "uefa.champions"),
]
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings"
ESPN_LEADERS = "https://site.web.api.espn.com/apis/site/v2/sports/soccer/{league}/statistics"
NHL_API = "https://api-web.nhle.com/v1"

NHL_TIE = 0.22        # part des matchs NHL à égalité après 60 minutes
NHL_TOTAL = 6.2       # buts par match NHL quand aucune ligne de buts n'est disponible
DAYS_BACK = 3         # nombre de jours passés relus pour compléter les résultats
POSITIONS = {"C": "Centre", "L": "Ailier", "R": "Ailier", "D": "Défenseur"}
HEADERS = {"User-Agent": "analyse-sport/1.0 (usage personnel)"}

NOTES = []


def note(msg):
    """Message affiché en haut du site (source indisponible, donnée manquante...)."""
    print(msg, file=sys.stderr)
    if msg not in NOTES:
        NOTES.append(msg)


# ---------------------------------------------------------------- accès réseau

def get_json(url, params=None):
    """GET avec 3 tentatives. Renvoie le JSON ou None, sans jamais lever d'exception."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            print(f"Réseau : {exc.__class__.__name__} sur {url}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        print(f"HTTP {r.status_code} sur {url}", file=sys.stderr)
        return None
    return None


# ---------------------------------------------------------------- outils

STOP_WORDS = {"fc", "cf", "ac", "as", "sc", "afc", "ssc", "club", "de", "sv", "vfb", "vfl", "tsg", "rc",
              "ogc", "stade", "olympique", "calcio", "bc", "ud", "rcd", "sd", "cd", "ca", "us", "hsc",
              "fk", "1899", "1909", "1913", "1846", "1907", "04", "05"}


def norm(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    for ch in "-.'&/()":
        s = s.replace(ch, " ")
    words = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(words) if words else s.strip()


def similarity(a, b):
    """Ressemblance entre deux noms d'équipe, de 0 à 1."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if short in long_ and len(short) / len(long_) >= 0.5:
        return 0.9
    return difflib.SequenceMatcher(None, a, b).ratio()


def parse_time(s):
    s = s.replace("Z", "+00:00")
    if len(s) == 22 and s[16] == "+":          # format ESPN "2026-10-11T15:30+00:00"
        s = s[:16] + ":00" + s[16:]
    return dt.datetime.fromisoformat(s)


def paris(t):
    return t.astimezone(PARIS)


def implied(american):
    """Cote américaine (+150, -200, EVEN) → probabilité implicite brute."""
    if american in (None, ""):
        return None
    s = str(american).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        o = 100
    else:
        try:
            o = float(s.replace("+", ""))
        except ValueError:
            return None
    if o == 0:
        return None
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def pct(x):
    return round(x * 100)


def plural(n, word, word_pl=None):
    return f"{n} {word if n <= 1 else (word_pl or word + 's')}"


# ---------------------------------------------------------------- modèles

def score_grid(lh, la, n=11):
    ph = [poisson.pmf(i, lh) for i in range(n)]
    pa = [poisson.pmf(j, la) for j in range(n)]
    return [[ph[i] * pa[j] for j in range(n)] for i in range(n)]


def outcome_probs(lh, la, n=11):
    """Victoire domicile, égalité, victoire extérieur selon deux lois de Poisson."""
    g = score_grid(lh, la, n)
    home = sum(g[i][j] for i in range(n) for j in range(i))
    away = sum(g[i][j] for i in range(n) for j in range(i + 1, n))
    return home, 1 - home - away, away


def foot_lambdas(p_home, p_away):
    """Buts attendus de chaque équipe qui reproduisent les probabilités des cotes."""
    def f(v):
        h, _, a = outcome_probs(abs(v[0]), abs(v[1]))
        return [h - p_home, a - p_away]
    lh, la = fsolve(f, [1.4, 1.1])
    return abs(lh), abs(la)


def table_lambdas(sh, sa, league_gpg):
    """Sans cotes : attaque et défense tirées du classement, ramenées vers la moyenne (petits échantillons)."""
    def rate(goals, games):
        return (goals + league_gpg * 6) / (games + 6)
    att_h, def_h = rate(sh["gf"], sh["gp"]) / league_gpg, rate(sh["ga"], sh["gp"]) / league_gpg
    att_a, def_a = rate(sa["gf"], sa["gp"]) / league_gpg, rate(sa["ga"], sa["gp"]) / league_gpg
    return league_gpg * att_h * def_a * 1.12, league_gpg * att_a * def_h * 0.88


def top_scores(lh, la, k=3, n=9, exclude_draws=False):
    g = score_grid(lh, la, n)
    cells = sorted(((g[i][j], f"{i}-{j}") for i in range(n) for j in range(n)
                    if not (exclude_draws and i == j)), reverse=True)[:k]
    return [c[1] for c in cells], [pct(c[0]) for c in cells]


def nhl_split(win_home, total):
    """Vainqueur final → (domicile 60 min, égalité 60 min, extérieur 60 min, buts attendus dom., ext.)."""
    lean = 0.5 + 0.5 * (win_home - 0.5)          # la prolongation est proche du pile ou face
    p_h = max(0.02, win_home - NHL_TIE * lean)
    p_a = max(0.02, (1 - win_home) - NHL_TIE * (1 - lean))
    p_d = max(0.0, 1 - p_h - p_a)
    target = p_h / p_a

    def f(s):
        h, _, a = outcome_probs(total * s, total * (1 - s), n=15)
        return h / a - target
    try:
        s = brentq(f, 0.12, 0.88)
    except ValueError:
        s = 0.5
    return p_h, p_d, p_a, total * s, total * (1 - s)


# ---------------------------------------------------------------- ESPN

def espn_events(sport, league, day):
    data = get_json(ESPN_SCOREBOARD.format(sport=sport, league=league), {"dates": day.strftime("%Y%m%d")})
    if data is None:
        return None
    return data.get("events", []) or []


def espn_odds(comp, three_way):
    """Cotes ESPN → probabilités (marge retirée), bookmaker et ligne de buts."""
    for o in comp.get("odds") or []:
        if not o:
            continue
        ml = o.get("moneyline") or {}

        def line(side):
            d = ml.get(side) or {}
            for k in ("close", "current", "open"):
                v = (d.get(k) or {}).get("odds")
                if v not in (None, ""):
                    return v
            return None
        home = line("home") or (o.get("homeTeamOdds") or {}).get("moneyLine")
        away = line("away") or (o.get("awayTeamOdds") or {}).get("moneyLine")
        values = [home, away]
        if three_way:
            draw = line("draw") or (o.get("drawOdds") or {}).get("moneyLine")
            values = [home, draw, away]
        probs = [implied(v) for v in values]
        if any(p is None for p in probs):
            continue
        s = sum(probs)
        return dict(p=[p / s for p in probs],
                    provider=((o.get("provider") or {}).get("name") or "un bookmaker"),
                    total=o.get("overUnder"))
    return None


def espn_standings(league):
    data = get_json(ESPN_STANDINGS.format(league=league)) or {}
    rows = {}
    groups = data.get("children") or [data]
    for grp in groups:
        for e in (grp.get("standings") or {}).get("entries", []):
            st = {s.get("name"): s.get("value") for s in e.get("stats", [])}
            rows[str(e["team"]["id"])] = dict(
                rank=int(st.get("rank") or 0), gp=int(st.get("gamesPlayed") or 0),
                w=int(st.get("wins") or 0), d=int(st.get("ties") or 0), l=int(st.get("losses") or 0),
                gf=int(st.get("pointsFor") or 0), ga=int(st.get("pointsAgainst") or 0),
                pts=int(st.get("points") or 0))
    return rows


def espn_leaders(league):
    """Meilleurs buteurs et passeurs du championnat, regroupés par équipe (identifiant ESPN)."""
    data = get_json(ESPN_LEADERS.format(league=league)) or {}
    players = {}
    for cat in data.get("stats", []):
        name = (cat.get("name") or "").lower()
        kind = "goals" if "goal" in name else ("assists" if "assist" in name else None)
        if not kind:
            continue
        for item in cat.get("leaders", []):
            a = item.get("athlete") or {}
            pid, team = a.get("id"), (a.get("team") or {}).get("id")
            if not pid or not team:
                continue
            p = players.setdefault(pid, dict(name=a.get("displayName", ""), team=str(team), goals=0, assists=0, apps=0))
            p[kind] = max(p[kind], int(item.get("value") or 0))
            for s in a.get("statistics") or []:
                sname, val = s.get("name"), int(s.get("value") or 0)
                if sname == "appearances":
                    p["apps"] = max(p["apps"], val)
                elif sname in ("totalGoals", "goals"):
                    p["goals"] = max(p["goals"], val)
                elif sname in ("goalAssists", "assists"):
                    p["assists"] = max(p["assists"], val)
    by_team = {}
    for p in players.values():
        by_team.setdefault(p["team"], []).append(p)
    return by_team


# ---------------------------------------------------------------- foot

FORM_FR = {"W": "V", "D": "N", "L": "D"}


def foot(today, history):
    rows = []
    failures = 0
    for league, code in LEAGUES:
        try:
            got = foot_league(today, history, league, code)
            if got is None:
                failures += 1
            else:
                rows += got
        except Exception as exc:          # une ligue en erreur ne doit pas bloquer les autres
            failures += 1
            note(f"{league} : erreur inattendue ({exc.__class__.__name__}).")
    if failures == len(LEAGUES):
        note("Foot : l'API d'ESPN ne répond pas ou a changé de format. Aucune donnée foot aujourd'hui.")
    rows.sort(key=lambda r: r["kickoff"])
    return rows


def foot_league(today, history, league, code):
    # 1. résultats des jours précédents
    for back in range(1, DAYS_BACK + 1):
        for e in espn_events("soccer", code, today - dt.timedelta(days=back)) or []:
            resolve_foot(e, history)
    # 2. matchs du jour
    events = espn_events("soccer", code, today)
    if events is None:
        return None
    for e in events:
        resolve_foot(e, history)
    todays = [e for e in events if paris(parse_time(e["date"])).date() == today]
    if not todays:
        return []
    table = espn_standings(code)
    gp = sum(r["gp"] for r in table.values())
    league_gpg = (sum(r["gf"] for r in table.values()) / gp) if gp else 1.4
    leaders = espn_leaders(code)
    if not leaders:
        note(f"{league} : buteurs indisponibles aujourd'hui.")
    return [foot_match(e, league, table, league_gpg or 1.4, leaders, history) for e in todays]


def competitors(e):
    comp = e["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    return comp, teams["home"], teams["away"]


def resolve_foot(e, history):
    key = f"foot-{e['id']}"
    if key not in history or history[key].get("result") is not None:
        return
    if not ((e.get("status") or {}).get("type") or {}).get("completed"):
        return
    _, h, a = competitors(e)
    try:
        sh, sa = int(h["score"]), int(a["score"])
    except (KeyError, TypeError, ValueError):
        return
    history[key]["score"] = f"{sh}-{sa}"
    history[key]["result"] = 0 if sh > sa else (1 if sh == sa else 2)


def foot_players(squad, team_goals, lam):
    rows = []
    for p in squad:
        g, a = p["goals"], p["assists"]
        share_goal = (g + 0.12 * 5) / (team_goals + 5)
        share_dec = min(1.5, (g + a + 0.25 * 5) / (team_goals + 5))
        rows.append(dict(name=p["name"], goals=g, assists=a,
                         p_goal=pct(1 - math.exp(-lam * share_goal * 0.85)),
                         p_dec=pct(1 - math.exp(-lam * share_dec * 0.85))))
    return rows


def foot_match(e, league, table, league_gpg, leaders, history):
    comp, H, A = competitors(e)
    kickoff = parse_time(e["date"])
    hid, aid = str(H["team"]["id"]), str(A["team"]["id"])
    home = H["team"].get("shortDisplayName") or H["team"].get("displayName")
    away = A["team"].get("shortDisplayName") or A["team"].get("displayName")
    sh, sa = table.get(hid), table.get(aid)
    state = ((e.get("status") or {}).get("type") or {}).get("state", "pre")
    key = f"foot-{e['id']}"
    saved = history.get(key, {}).get("p")
    odds = espn_odds(comp, three_way=True)
    if state != "pre" and saved and len(saved) == 3:
        # après le coup d'envoi, ESPN affiche des cotes en direct : on garde celles d'avant-match
        p_h, p_d, p_a = saved
        lh, la = foot_lambdas(p_h, p_a)
        source = history[key].get("source", "Cotes d'avant-match")
    elif odds and state == "pre":
        p_h, p_d, p_a = odds["p"]
        lh, la = foot_lambdas(p_h, p_a)
        source = f"Cotes {odds['provider']}"
    elif sh and sa:
        lh, la = table_lambdas(sh, sa, league_gpg)
        p_h, p_d, p_a = outcome_probs(lh, la)
        source = ("Modèle maison fondé sur le classement (pas de cotes disponibles)" if state == "pre"
                  else "Cotes d'avant-match non archivées : estimation fondée sur le classement")
    else:
        lh, la = 1.45, 1.15
        p_h, p_d, p_a = outcome_probs(lh, la)
        source = "Estimation par défaut (ni cotes ni classement)"
    scores, scorep = top_scores(lh, la)
    g = score_grid(lh, la)
    over25 = 1 - sum(g[i][j] for i in range(3) for j in range(3) if i + j <= 2)
    btts = (1 - math.exp(-lh)) * (1 - math.exp(-la))

    players, deci = [], []
    for side, tid, st, lam, name in (("h", hid, sh, lh, home), ("a", aid, sa, la, away)):
        rows = foot_players(leaders.get(tid, []), max((st or {}).get("gf", 0), 1), lam)
        for r in rows:
            r.update(side=side, team=name)
        players += sorted(rows, key=lambda r: -r["p_goal"])[:4]
        deci += sorted(rows, key=lambda r: -r["p_dec"])[:3]
    players.sort(key=lambda r: -r["p_goal"])
    deci.sort(key=lambda r: -r["p_dec"])

    if state == "pre":
        old = history.get(key, {})
        history[key] = dict(sport="foot", league=league, date=paris(kickoff).date().isoformat(),
                            home=home, away=away, p=[round(p_h, 4), round(p_d, 4), round(p_a, 4)],
                            source=source, result=old.get("result"), score=old.get("score"))

    ctx = []
    for name, st in ((home, sh), (away, sa)):
        if st:
            ctx.append(f"{name} : {st['rank']}e, {plural(st['pts'], 'point')}, "
                       f"{st['gf']} buts marqués et {st['ga']} encaissés en {plural(st['gp'], 'match', 'matchs')}.")
    if not leaders.get(hid) and not leaders.get(aid):
        ctx.append("Aucun joueur de ces deux équipes dans le top 50 des buteurs et passeurs du championnat.")

    fav, pf = (home, p_h) if p_h >= p_a else (away, p_a)
    if pf >= 0.6:
        reading = f"{fav} est nettement favori ({pct(pf)} %)."
    elif pf >= 0.45:
        reading = f"{fav} est favori sans écraser le match ({pct(pf)} %)."
    else:
        reading = "Match très ouvert : aucune équipe ne dépasse 45 %."
    if p_d >= 0.29:
        reading += f" Le nul est une issue sérieuse ({pct(p_d)} %)."

    def form(c):
        return "".join(FORM_FR.get(ch, "") for ch in (c.get("form") or "")[-5:])

    return dict(
        id=key, sport="foot", league=league, kickoff=kickoff.isoformat(), time=paris(kickoff).strftime("%H:%M"),
        state=state, score=(f"{H.get('score')}-{A.get('score')}" if state != "pre" else None),
        home=home, away=away, form_h=form(H), form_a=form(A),
        rank_h=(sh or {}).get("rank"), rank_a=(sa or {}).get("rank"),
        pH=pct(p_h), pD=pct(p_d), pA=pct(p_a), source=source,
        xgH=round(lh, 2), xgA=round(la, 2), scores=scores, scorep=scorep,
        over25=pct(over25), btts=pct(btts), players=players, deci=deci, ctx=ctx, reading=reading)


# ---------------------------------------------------------------- NHL

def nhl_get(path):
    return get_json(f"{NHL_API}{path}")


def toi_minutes(toi):
    try:
        m, s = str(toi).split(":")
        return int(m) + int(s) / 60
    except ValueError:
        return 0.0


def last_lineup(abbr):
    """Joueurs alignés lors du dernier match de l'équipe (sert à écarter les blessés) et gardien titulaire."""
    sched = nhl_get(f"/club-schedule-season/{abbr}/now") or {}
    done = [g for g in sched.get("games", []) if g.get("gameState") in ("OFF", "FINAL")]
    if not done:
        return None
    game = max(done, key=lambda g: g.get("startTimeUTC", ""))
    box = nhl_get(f"/gamecenter/{game['id']}/boxscore") or {}
    side = "homeTeam" if (box.get("homeTeam") or {}).get("abbrev") == abbr else "awayTeam"
    stats = (box.get("playerByGameStats") or {}).get(side) or {}
    skaters = {p["playerId"] for grp in ("forwards", "defense") for p in stats.get(grp, [])}
    goalies = [p for p in stats.get("goalies", []) if toi_minutes(p.get("toi")) > 0]
    starter = next((p for p in goalies if p.get("starter")), None) or \
        max(goalies, key=lambda p: toi_minutes(p.get("toi")), default=None)
    # deux gardiens ayant joué au moins 10 minutes : le titulaire a été remplacé (blessure ou contre-performance)
    relieved = None
    if len([p for p in goalies if toi_minutes(p.get("toi")) >= 10]) >= 2 and starter and starter.get("starter"):
        relieved = starter["playerId"]
    return dict(skaters=skaters, starter=(starter or {}).get("playerId"), relieved=relieved,
                goalies_dressed={p["playerId"] for p in stats.get("goalies", [])},
                date=game.get("gameDate"))


def team_profile(abbr, season):
    cur = nhl_get(f"/club-stats/{abbr}/{season}/2") or {}
    prev = nhl_get(f"/club-stats/{abbr}/{season - 10001}/2") or {}
    prev_by_id = {p["playerId"]: p for p in prev.get("skaters", [])}
    last = last_lineup(abbr)
    dressed = last["skaters"] if last and last["skaters"] else None
    skaters = []
    for p in cur.get("skaters", []):
        if dressed is not None and p["playerId"] not in dressed:
            continue
        q = prev_by_id.get(p["playerId"], {})
        gp = p.get("gamesPlayed", 0) + q.get("gamesPlayed", 0)
        g = p.get("goals", 0) + q.get("goals", 0)
        a = p.get("assists", 0) + q.get("assists", 0)
        cg, ca, cgp = p.get("goals", 0), p.get("assists", 0), p.get("gamesPlayed", 0)
        skaters.append(dict(
            name=f"{p['firstName']['default']} {p['lastName']['default']}",
            pos=POSITIONS.get(p.get("positionCode"), ""),
            cur=f"{plural(cg, 'but')}, {plural(ca, 'passe')} en {cgp}",
            gpg=(g + 0.10 * 20) / (gp + 20), ppg=(g + a + 0.35 * 20) / (gp + 20),
            apg=(a + 0.20 * 20) / (gp + 20), new=not q))
    return dict(skaters=skaters, goalies=cur.get("goalies", []),
                prev_goals=sum(p.get("goals", 0) for p in prev.get("skaters", [])),
                has_prev=bool(prev.get("skaters")), last=last)


def projected_goalie(profile, back_to_back):
    goalies = [g for g in profile["goalies"] if g.get("gamesStarted", 0) > 0] or profile["goalies"]
    if not goalies:
        return None
    # un gardien absent de la feuille du dernier match est probablement blessé
    dressed = (profile["last"] or {}).get("goalies_dressed") or set()
    pool = [g for g in goalies if g["playerId"] in dressed] or goalies
    goalies = pool
    relieved = (profile["last"] or {}).get("relieved")
    note_relief = ""
    if relieved:
        rest = [g for g in goalies if g["playerId"] != relieved]
        if rest:
            goalies, note_relief = rest, "le titulaire a été remplacé en cours de match la dernière fois (blessure possible)"
    starter = (profile["last"] or {}).get("starter")
    pool = goalies
    if back_to_back and starter:
        pool = [g for g in goalies if g["playerId"] != starter] or goalies
    best = max(pool, key=lambda g: (g.get("gamesStarted", 0), g.get("savePercentage") or 0))
    sv = best.get("savePercentage")
    return dict(name=f"{best['firstName']['default']} {best['lastName']['default']}",
                starts=best.get("gamesStarted", 0),
                sv=(f"{sv * 100:.1f}".replace(".", ",") + " % d'arrêts") if sv else "pas encore joué",
                note=note_relief or ("le titulaire d'hier devrait souffler" if back_to_back and starter else ""))


def streak_text(st):
    code, n = st.get("streakCode"), st.get("streakCount") or 0
    if not code or not n:
        return ""
    if code == "W":
        return f"{plural(n, 'victoire')} de suite"
    if code == "L":
        return f"{plural(n, 'défaite')} de suite"
    return f"{plural(n, 'défaite')} en prolongation de suite"


def nhl(today, history):
    # 1. résultats des jours précédents
    yesterday_teams = set()
    for back in range(1, DAYS_BACK + 2):
        day = today - dt.timedelta(days=back)
        games = (nhl_get(f"/score/{day.isoformat()}") or {}).get("games", [])
        for g in games:
            if back == 1:
                yesterday_teams |= {g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]}
            resolve_nhl(g, history)
    # 2. matchs du jour
    sched = nhl_get(f"/schedule/{today.isoformat()}")
    if sched is None:
        note("NHL : l'API officielle ne répond pas. Aucune donnée NHL aujourd'hui.")
        return []
    day = next((d for d in sched.get("gameWeek", []) if d.get("date") == today.isoformat()), None)
    games = (day or {}).get("games", [])
    if not games:
        return []
    standings = {s["teamAbbrev"]["default"]: s
                 for s in (nhl_get("/standings/now") or {}).get("standings", [])}
    espn = espn_events("hockey", "nhl", today) or []
    if not espn:
        note("NHL : cotes ESPN indisponibles, probabilités tirées du classement (moins fiables).")
    rows = []
    for g in games:
        try:
            rows.append(nhl_game(g, standings, espn, yesterday_teams, history))
        except Exception as exc:
            note(f"NHL {g['awayTeam']['abbrev']} contre {g['homeTeam']['abbrev']} : "
                 f"erreur inattendue ({exc.__class__.__name__}).")
    rows.sort(key=lambda r: r["kickoff"])
    return rows


def resolve_nhl(g, history):
    key = f"nhl-{g['id']}"
    if key not in history or history[key].get("result") is not None:
        return
    if g.get("gameState") not in ("OFF", "FINAL"):
        return
    hs, as_ = g["homeTeam"].get("score"), g["awayTeam"].get("score")
    if hs is None or as_ is None:
        return
    period = (g.get("gameOutcome") or {}).get("lastPeriodType", "REG")
    suffix = {"OT": " (prol.)", "SO": " (tirs au but)"}.get(period, "")
    history[key]["score"] = f"{hs}-{as_}{suffix}"
    history[key]["result"] = 0 if hs > as_ else 1


def nhl_game(g, standings, espn, yesterday_teams, history):
    season = int(g["season"])
    H, A = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
    sh, sa = standings.get(H, {}), standings.get(A, {})
    hname = (sh.get("teamName") or {}).get("default") or H
    aname = (sa.get("teamName") or {}).get("default") or A
    start = parse_time(g["startTimeUTC"])

    odds = None
    for e in espn:
        comp, eh, ea = competitors(e)
        if similarity(hname, eh["team"].get("displayName", "")) >= 0.8 and \
                similarity(aname, ea["team"].get("displayName", "")) >= 0.8:
            odds = espn_odds(comp, three_way=False)
            break
    total = NHL_TOTAL
    started = g.get("gameState", "FUT") not in ("FUT", "PRE")
    saved = history.get(f"nhl-{g['id']}", {})
    if started and saved.get("p"):
        # match commencé : on garde les probabilités d'avant-match, pas les cotes en direct
        odds = None
        win_h = saved["p"][0]
        source = saved.get("source", "Cotes d'avant-match")
    elif started:
        odds = None
    if started and saved.get("p"):
        pass
    elif odds:
        win_h = odds["p"][0]
        source = f"Cotes {odds['provider']} (vainqueur final)"
        try:
            if odds.get("total"):
                total = max(4.5, float(odds["total"]) - 0.2)
        except (TypeError, ValueError):
            pass
    else:
        ph_, pa_ = sh.get("pointPctg") or 0.5, sa.get("pointPctg") or 0.5
        win_h = min(0.75, max(0.25, 0.5 + (ph_ - pa_) * 0.9 + 0.04))
        source = "Modèle maison fondé sur le classement (pas de cotes disponibles)"
    win_a = 1 - win_h
    p_h, p_d, p_a, lh, la = nhl_split(win_h, total)
    scores, scorep = top_scores(lh, la, n=10, exclude_draws=True)

    b2b_h, b2b_a = H in yesterday_teams, A in yesterday_teams
    prof_h, prof_a = team_profile(H, season), team_profile(A, season)

    def ref(st, prof):
        gf, gp = st.get("goalFor", 0), st.get("gamesPlayed", 0)
        if prof["has_prev"]:
            return (gf + prof["prev_goals"]) / (gp + 82)
        return gf / gp if gp else total / 2

    players, deci = [], []
    for side, prof, lam, st, name in (("h", prof_h, lh, sh, hname), ("a", prof_a, la, sa, aname)):
        k = lam / max(ref(st, prof), 1.5)
        rows = []
        for p in prof["skaters"]:
            rows.append(dict(name=p["name"], pos=p["pos"], cur=p["cur"], new=p["new"], side=side, team=name,
                             p_goal=pct(1 - math.exp(-p["gpg"] * k)),
                             p_point=pct(1 - math.exp(-p["ppg"] * k)),
                             p_assist=pct(1 - math.exp(-p["apg"] * k))))
        players += sorted(rows, key=lambda r: -r["p_goal"])[:4]
        deci += sorted(rows, key=lambda r: -r["p_point"])[:4]
    players.sort(key=lambda r: -r["p_goal"])
    deci.sort(key=lambda r: -r["p_point"])

    state = g.get("gameState", "FUT")
    key = f"nhl-{g['id']}"
    if state in ("FUT", "PRE"):
        old = history.get(key, {})
        history[key] = dict(sport="nhl", league="NHL", date=paris(start).date().isoformat(),
                            home=hname, away=aname, p=[round(win_h, 4), round(win_a, 4)],
                            source=source, result=old.get("result"), score=old.get("score"))

    def record(st):
        if not st:
            return ""
        txt = f"{st.get('wins', 0)}-{st.get('losses', 0)}-{st.get('otLosses', 0)} · " \
              f"{st.get('divisionSequence', '?')}e {st.get('divisionName', '')}"
        streak = streak_text(st)
        return txt + (f" · {streak}" if streak else "")

    ctx = []
    for name, st, b2b in ((hname, sh, b2b_h), (aname, sa, b2b_a)):
        if st and st.get("gamesPlayed"):
            ctx.append(f"{name} : {st.get('goalFor', 0)} buts marqués et {st.get('goalAgainst', 0)} encaissés "
                       f"en {plural(st['gamesPlayed'], 'match', 'matchs')}.")
        if b2b:
            ctx.append(f"{name} a joué hier soir : deuxième match en deux soirs, fatigue probable.")
    if not b2b_h and not b2b_a:
        ctx.append("Aucune des deux équipes n'a joué hier.")
    ctx.append("Blessés : non vérifiés automatiquement. Seuls les joueurs alignés au dernier match sont retenus.")

    fav, pf = (hname, win_h) if win_h >= win_a else (aname, win_a)
    if pf >= 0.65:
        reading = f"{fav} est nettement favori ({pct(pf)} % en comptant la prolongation)."
    elif pf >= 0.55:
        reading = f"{fav} est favori ({pct(pf)} % en comptant la prolongation)."
    else:
        reading = "Match très équilibré : au hockey, c'est presque un pile ou face."

    live = state in ("LIVE", "CRIT", "OFF", "FINAL")
    return dict(
        id=key, sport="nhl", league="NHL", kickoff=start.isoformat(), time=paris(start).strftime("%H:%M"),
        state=state, score=(f"{g['homeTeam'].get('score', 0)}-{g['awayTeam'].get('score', 0)}" if live else None),
        home=hname, away=aname, rec_h=record(sh), rec_a=record(sa),
        winH=pct(win_h), winA=pct(win_a), pH=pct(p_h), pD=pct(p_d), pA=pct(p_a), source=source,
        xgH=round(lh, 2), xgA=round(la, 2), total=round(total, 1), scores=scores, scorep=scorep,
        goalie_h=projected_goalie(prof_h, b2b_h), goalie_a=projected_goalie(prof_a, b2b_a),
        players=players, deci=deci, ctx=ctx, reading=reading)


# ---------------------------------------------------------------- bilan

def bilan(history):
    done = [h for h in history.values() if h.get("result") is not None and h.get("p")]
    out = {}
    for sport in ("foot", "nhl"):
        rows = [h for h in done if h["sport"] == sport]
        if not rows:
            out[sport] = dict(n=0)
            continue
        hits, brier, bins = 0, 0.0, {}
        for h in rows:
            p, r = h["p"], h["result"]
            fav = max(range(len(p)), key=lambda i: p[i])
            hits += fav == r
            brier += sum((p[i] - (1 if i == r else 0)) ** 2 for i in range(len(p)))
            lo = min(int(p[fav] * 10), 9) * 10
            b = bins.setdefault(lo, [0, 0, 0.0])
            b[0] += 1
            b[1] += fav == r
            b[2] += p[fav]
        n = len(rows)
        out[sport] = dict(
            n=n, fav_rate=pct(hits / n), fav_expected=pct(sum(max(h["p"]) for h in rows) / n),
            brier=round(brier / n, 3), brier_naive=0.667 if sport == "foot" else 0.5,
            bins=[dict(range=f"{lo}-{lo + 10} %", n=v[0], announced=pct(v[2] / v[0]), actual=pct(v[1] / v[0]))
                  for lo, v in sorted(bins.items())])
    recent = sorted(done, key=lambda h: h["date"], reverse=True)[:40]
    out["recent"] = []
    for h in recent:
        p = h["p"]
        fav = max(range(len(p)), key=lambda i: p[i])
        labels = [h["home"], "Nul", h["away"]] if len(p) == 3 else [h["home"], h["away"]]
        out["recent"].append(dict(date=h["date"], sport=h["sport"], league=h["league"], home=h["home"],
                                  away=h["away"], score=h.get("score"), fav=labels[fav], favp=pct(p[fav]),
                                  ok=fav == h["result"]))
    return out


# ---------------------------------------------------------------- programme principal

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main(now=None):
    now = now or dt.datetime.now(PARIS)
    today = now.date()
    history = load_json(HIST_FILE, {})
    foot_rows = foot(today, history)
    nhl_rows = nhl(today, history)
    cutoff = (today - dt.timedelta(days=400)).isoformat()
    history = {k: v for k, v in history.items() if v.get("date", "") >= cutoff}
    data = dict(generated=now.isoformat(timespec="minutes"), date=today.isoformat(),
                foot=foot_rows, nhl=nhl_rows, bilan=bilan(history), notes=NOTES[-12:])
    save_json(HIST_FILE, history)
    save_json(DATA_FILE, data)
    print(f"OK : {len(foot_rows)} matchs de foot, {len(nhl_rows)} matchs NHL, "
          f"{sum(1 for h in history.values() if h.get('result') is not None)} résultats archivés.")
    return data


if __name__ == "__main__":
    main()
