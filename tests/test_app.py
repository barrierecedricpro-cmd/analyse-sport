"""Tests de bout en bout sans réseau : les réponses des API sont simulées.

Lancer : python -m unittest discover -s tests -v
"""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402

NOW = dt.datetime(2026, 10, 11, 14, 0, tzinfo=app.PARIS)


def skater(pid, first, last, pos, gp, g, a):
    return dict(playerId=pid, firstName={"default": first}, lastName={"default": last},
                positionCode=pos, gamesPlayed=gp, goals=g, assists=a)


def goalie(pid, first, last, gs, sv):
    return dict(playerId=pid, firstName={"default": first}, lastName={"default": last},
                gamesPlayed=gs, gamesStarted=gs, savePercentage=sv)


CLUB = {
    ("WSH", 20262027): dict(skaters=[skater(1, "Tom", "Wilson", "R", 4, 2, 3), skater(2, "Jakob", "Chychrun", "D", 4, 1, 2),
                                     skater(3, "Blessé", "Absent", "C", 2, 1, 0)],
                            goalies=[goalie(10, "Logan", "Thompson", 3, .929), goalie(11, "Charlie", "Lindgren", 1, .931)]),
    ("WSH", 20252026): dict(skaters=[skater(1, "Tom", "Wilson", "R", 72, 30, 32), skater(2, "Jakob", "Chychrun", "D", 80, 26, 34)]),
    ("SEA", 20262027): dict(skaters=[skater(4, "Jared", "McCann", "L", 5, 1, 5), skater(5, "Mackie", "Samoskevich", "R", 4, 3, 0)],
                            goalies=[goalie(20, "Philipp", "Grubauer", 2, .917), goalie(21, "Joey", "Daccord", 3, .911)]),
    ("SEA", 20252026): dict(skaters=[skater(4, "Jared", "McCann", "L", 52, 20, 20)]),
}


def espn_soccer_event(eid, date, state, completed, home, away, odds=None):
    def comp(side, t):
        return dict(homeAway=side, team=dict(id=t[0], displayName=t[1], shortDisplayName=t[1]),
                    score=t[2], form="WDWWL")
    return dict(id=eid, date=date, status=dict(type=dict(state=state, completed=completed)),
                competitions=[dict(competitors=[comp("home", home), comp("away", away)], odds=odds or [])])


def fake_get_json(url, params=None):
    p = params or {}
    if "soccer/eng.1/scoreboard" in url:
        if p.get("dates") == "20261011":
            odds = [None, dict(provider=dict(name="ESPN BET"), overUnder=2.5,
                               moneyline=dict(home=dict(close=dict(odds="+230")), away=dict(close=dict(odds="+105"))),
                               drawOdds=dict(moneyLine=275))]
            return dict(events=[espn_soccer_event("401", "2026-10-11T15:30Z", "pre", False,
                                                  ("364", "Liverpool", "0"), ("382", "Man City", "0"), odds)])
        if p.get("dates") == "20261010":
            return dict(events=[espn_soccer_event("900", "2026-10-10T14:00Z", "post", True,
                                                  ("359", "Arsenal", "2"), ("363", "Chelsea", "1"))])
        return dict(events=[])
    if "/scoreboard" in url and "soccer" in url:
        return dict(events=[])
    if "soccer/eng.1/standings" in url:
        def entry(tid, rank, gp, gf, ga, pts):
            return dict(team=dict(id=tid), stats=[dict(name="rank", value=rank), dict(name="gamesPlayed", value=gp),
                                                  dict(name="pointsFor", value=gf), dict(name="pointsAgainst", value=ga),
                                                  dict(name="points", value=pts)])
        return dict(children=[dict(standings=dict(entries=[entry("382", 1, 5, 13, 5, 15), entry("364", 7, 5, 7, 4, 9)]))])
    if "soccer/eng.1/statistics" in url:
        def lead(pid, name, tid, val):
            return dict(value=val, athlete=dict(id=pid, displayName=name, team=dict(id=tid)))
        return dict(stats=[dict(name="goalsLeaders", leaders=[lead("a1", "Erling Haaland", "382", 5), lead("a2", "Alexander Isak", "364", 4)]),
                           dict(name="assistsLeaders", leaders=[lead("a3", "Antoine Semenyo", "382", 3)])])
    if "hockey/nhl/scoreboard" in url:
        comp = dict(competitors=[dict(homeAway="home", team=dict(displayName="Washington Capitals")),
                                 dict(homeAway="away", team=dict(displayName="Seattle Kraken"))],
                    odds=[dict(provider=dict(name="ESPN BET"), overUnder=6.5,
                               moneyline=dict(home=dict(close=dict(odds="-218")), away=dict(close=dict(odds="+180"))))])
        return dict(events=[dict(id="e1", date="2026-10-11T21:00Z", competitions=[comp])])
    if url.endswith("/score/2026-10-10"):
        return dict(games=[dict(id=777, gameState="OFF", homeTeam=dict(abbrev="BOS", score=5),
                                awayTeam=dict(abbrev="PHI", score=1), gameOutcome=dict(lastPeriodType="REG")),
                           dict(id=778, gameState="OFF", homeTeam=dict(abbrev="CHI", score=2),
                                awayTeam=dict(abbrev="SEA", score=5), gameOutcome=dict(lastPeriodType="REG"))])
    if "/score/" in url:
        return dict(games=[])
    if url.endswith("/schedule/2026-10-11"):
        return dict(gameWeek=[dict(date="2026-10-11", games=[dict(
            id=2026020084, season=20262027, startTimeUTC="2026-10-11T21:00:00Z", gameState="FUT",
            homeTeam=dict(abbrev="WSH"), awayTeam=dict(abbrev="SEA"))])])
    if url.endswith("/standings/now"):
        def st(abbr, name, gp, w, l, otl, gf, ga, seq, div, code, n):
            return dict(teamAbbrev={"default": abbr}, teamName={"default": name}, gamesPlayed=gp, wins=w, losses=l,
                        otLosses=otl, goalFor=gf, goalAgainst=ga, divisionSequence=seq, divisionName=div,
                        streakCode=code, streakCount=n, pointPctg=(2 * w + otl) / (2 * gp))
        return dict(standings=[st("WSH", "Washington Capitals", 4, 3, 1, 0, 14, 9, 4, "Metropolitan", "W", 2),
                               st("SEA", "Seattle Kraken", 5, 3, 2, 0, 21, 14, 4, "Pacific", "W", 1)])
    if "/club-stats/" in url:
        parts = url.split("/club-stats/")[1].split("/")
        return CLUB.get((parts[0], int(parts[1])), dict(skaters=[], goalies=[]))
    if "/club-schedule-season/" in url:
        abbr = url.split("/club-schedule-season/")[1].split("/")[0]
        gid = 55 if abbr == "WSH" else 56
        return dict(games=[dict(id=gid, gameState="OFF", startTimeUTC="2026-10-10T23:00:00Z", gameDate="2026-10-10")])
    if "/gamecenter/55/boxscore" in url:
        return dict(homeTeam=dict(abbrev="WSH"), playerByGameStats=dict(homeTeam=dict(
            forwards=[dict(playerId=1)], defense=[dict(playerId=2)], goalies=[dict(playerId=10, toi="60:00")])))
    if "/gamecenter/56/boxscore" in url:
        return dict(homeTeam=dict(abbrev="CHI"), playerByGameStats=dict(awayTeam=dict(
            forwards=[dict(playerId=4), dict(playerId=5)], defense=[], goalies=[dict(playerId=21, toi="59:10")])))
    return None


class AppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        app.DATA_FILE = os.path.join(self.tmp, "data.json")
        app.HIST_FILE = os.path.join(self.tmp, "history.json")
        app.NOTES.clear()
        app.get_json = fake_get_json
        app.time.sleep = lambda s: None
        history = {"foot-900": dict(sport="foot", league="Premier League", date="2026-10-10", home="Arsenal",
                                    away="Chelsea", p=[0.5, 0.27, 0.23], result=None, score=None),
                   "nhl-777": dict(sport="nhl", league="NHL", date="2026-10-10", home="Boston Bruins",
                                   away="Philadelphia Flyers", p=[0.62, 0.38], result=None, score=None)}
        app.save_json(app.HIST_FILE, history)
        self.data = app.main(NOW)

    def test_foot_probabilities_follow_odds(self):
        m = self.data["foot"][0]
        raw = [app.implied("+230"), app.implied(275), app.implied("+105")]
        fair = [x / sum(raw) for x in raw]
        self.assertEqual([m["pH"], m["pD"], m["pA"]], [round(x * 100) for x in fair])
        h, d, a = app.outcome_probs(m["xgH"], m["xgA"])
        self.assertAlmostEqual(h, fair[0], delta=0.01)
        self.assertAlmostEqual(a, fair[2], delta=0.01)
        self.assertEqual(m["players"][0]["name"], "Erling Haaland")
        self.assertTrue(all(0 < p["p_goal"] < 100 for p in m["players"]))
        self.assertIn("Semenyo", " ".join(p["name"] for p in m["deci"]))

    def test_nhl_game(self):
        g = self.data["nhl"][0]
        raw = [app.implied("-218"), app.implied("+180")]
        self.assertEqual(g["winH"], round(raw[0] / sum(raw) * 100))
        self.assertAlmostEqual(g["xgH"] + g["xgA"], 6.3, delta=0.01)
        names = [p["name"] for p in g["players"]]
        self.assertNotIn("Blessé Absent", names)          # pas aligné au dernier match
        self.assertEqual(g["goalie_a"]["name"], "Philipp Grubauer")   # Seattle a joué hier, Daccord souffle
        self.assertEqual(g["goalie_h"]["name"], "Logan Thompson")
        self.assertTrue(any("joué hier" in c for c in g["ctx"]))
        self.assertTrue(g["deci"][0]["p_point"] >= g["deci"][-1]["p_point"])

    def test_history_and_bilan(self):
        hist = app.load_json(app.HIST_FILE, {})
        self.assertEqual(hist["foot-900"]["result"], 0)
        self.assertEqual(hist["foot-900"]["score"], "2-1")
        self.assertEqual(hist["nhl-777"]["result"], 0)
        self.assertIn("foot-401", hist)
        self.assertIn("nhl-2026020084", hist)
        self.assertEqual(self.data["bilan"]["foot"]["n"], 1)
        self.assertEqual(self.data["bilan"]["foot"]["fav_rate"], 100)
        self.assertEqual(self.data["bilan"]["nhl"]["n"], 1)

    def test_name_matching(self):
        self.assertLess(app.similarity("Paris Saint-Germain", "Paris FC"), 0.8)
        self.assertGreaterEqual(app.similarity("Olympique de Marseille", "Marseille"), 0.9)
        self.assertGreaterEqual(app.similarity("Washington Capitals", "Washington Capitals"), 1.0)

    def test_data_file_written(self):
        with open(app.DATA_FILE, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["date"], "2026-10-11")
        self.assertEqual(len(data["foot"]), 1)
        self.assertEqual(len(data["nhl"]), 1)


if __name__ == "__main__":
    unittest.main()
