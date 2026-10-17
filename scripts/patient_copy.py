"""Plain-language patient copy in English, Spanish and Haitian Creole (Kreyòl) for the three actions,
the care-team line, the low-sugar warning and a help route. Drafted by the team, not certified:
have a native speaker review before clinical use. Only fixed rule numbers appear (no measurements)."""

LANGS = ("en", "es", "ht")
ALLOWED_NUMBERS = {4, 8, 10, 13, 15, 20, 211, 911}

TEXT = {
    "walk": {"en": "Walk 10 to 20 minutes after dinner. If you take insulin or a sulfonylurea, ask your care team about low sugar first.",
             "es": "Camine de 10 a 20 minutos después de la cena. Si usa insulina o una sulfonilurea, pregunte primero a su equipo médico sobre el azúcar baja.",
             "ht": "Mache 10 a 20 minit apre manje aswè. Si ou pran ensilin oswa yon silfonilire, mande ekip swen ou sou sik ki ba anvan."},
    "walk_low": {"en": "Ask your care team before walking more, because of your low readings. Then walk 10 to 20 minutes after dinner and carry juice or glucose tablets.",
                 "es": "Por sus niveles bajos de azúcar, consulte a su equipo médico antes de caminar más. Luego camine de 10 a 20 minutos después de la cena y lleve jugo o tabletas de glucosa.",
                 "ht": "Paske sik ou konn desann twò ba, mande ekip swen ou anvan ou mache plis. Apre sa, mache 10 a 20 minit apre manje aswè, epi pote ji oswa grenn glikoz avè ou."},
    "late_earlier": {"en": "Try to start dinner before 8 pm.",
                     "es": "Trate de empezar a cenar antes de las 8 de la noche.",
                     "ht": "Eseye kòmanse manje aswè anvan 8 è di swa."},
    "late_keep": {"en": "Keep eating dinner before 8 pm.",
                  "es": "Siga cenando antes de las 8 de la noche.",
                  "ht": "Kontinye manje aswè anvan 8 è di swa."},
    "late_shift": {"en": "Eat your main meal at the same point in your waking day.",
                   "es": "Coma su comida principal a la misma hora de su día despierto.",
                   "ht": "Manje gwo repa ou a menm moman chak jou nan lè ou reveye."},
    "fast13": {"en": "Aim for 13 hours without food overnight.",
               "es": "Intente pasar 13 horas sin comer durante la noche.",
               "ht": "Eseye pase 13 è san manje pandan lannwit."},
    "fast_hold": {"en": "Don't make your overnight fast longer for now; talk to your care team first.",
                  "es": "Por ahora no alargue su ayuno de la noche; hable primero con su equipo médico.",
                  "ht": "Pou kounye a, pa fè jèn lannwit ou pi long; pale ak ekip swen ou anvan."},
    "team_lows": {"en": "Today: tell your care team about your low blood sugar readings. Ask if a diabetes medicine needs changing, and don't change your exercise or meal times until then.",
                  "es": "Hoy: avise a su equipo médico sobre sus niveles bajos de azúcar. Pregunte si hay que cambiar algún medicamento para la diabetes, y no cambie su ejercicio ni sus horarios de comida hasta entonces.",
                  "ht": "Jodi a: di ekip swen ou sou sik ki desann twò ba. Mande si yo dwe chanje yon medikaman pou dyabèt, epi pa chanje egzèsis ou ni lè ou manje jiska lè sa."},
    "team_high": {"en": "Today: send your blood sugar readings to your care team. Your average is very high.",
                  "es": "Hoy: envíe sus lecturas de azúcar a su equipo médico. Su promedio está muy alto.",
                  "ht": "Jodi a: voye chif sik ou yo bay ekip swen ou. Mwayèn sik ou twò wo."},
    "team_often_high": {"en": "Today: send your blood sugar readings to your care team. Your sugar is often very high.",
                        "es": "Hoy: envíe sus lecturas de azúcar a su equipo médico. Su azúcar está muy alto con frecuencia.",
                        "ht": "Jodi a: voye chif sik ou yo bay ekip swen ou. Sik ou souvan twò wo."},
    "team_alert": {"en": "Share these readings with your care team at your next visit.",
                   "es": "Muestre estas lecturas a su equipo médico en su próxima cita.",
                   "ht": "Montre ekip swen ou chif sa yo nan pwochen randevou ou."},
    "hypo": {"en": "Low sugar: if you feel shaky, sweaty or confused, eat or drink 15 grams of fast sugar (half a cup of juice or 4 glucose tablets), wait 15 minutes and check again. If someone passes out, call 911.",
             "es": "Azúcar baja: si se siente tembloroso, sudoroso o confundido, coma o beba 15 gramos de azúcar rápida (media taza de jugo o 4 tabletas de glucosa), espere 15 minutos y vuelva a medirse. Si alguien se desmaya, llame al 911.",
             "ht": "Sik ba: si ou santi ou ap tranble, ou ap swe oswa ou konfonn, manje oswa bwè 15 gram sik rapid (yon mwatye tas ji oswa 4 grenn glikoz), tann 15 minit epi tcheke ankò. Si yon moun endispoze, rele 911."},
    "help": {"en": "Need help with food, money or transport? Call 211 (free, any time) or ask your coach to connect you with a social worker.",
             "es": "¿Necesita ayuda con comida, dinero o transporte? Llame al 211 (gratis, a cualquier hora) o pida a su coach que le conecte con un trabajador social.",
             "ht": "Ou bezwen èd pou manje, lajan oswa transpò? Rele 211 (gratis, nenpòt lè) oswa mande antrenè ou mete ou an kontak ak yon travayè sosyal."},
}

# recommendation title -> action key
ACTION_FOR_TITLE = {
    "Take a short walk after meals": "walk", "Try a 10 to 20 minute walk after dinner": "walk",
    "Move dinner earlier": "late_earlier", "Keep dinner early": "late_keep",
    "Keep meal times steady around your shifts": "late_shift",
    "Aim for a 13-hour overnight break": "fast13", "Hold off on longer fasts for now": "fast_hold",
}

# English caution appended to walk advice for people with frequent lows (no numbers, so nothing to trace)
MED_CAUTION_CLINICIAN = ("Medicines are not recorded in this data. If the survivor takes insulin or a sulfonylurea (glipizide, glyburide, glimepiride), "
                         "ask their care team about lows before adding post-meal walks. The coach does not change medication.")
MED_CAUTION_PATIENT = "If you take insulin or a sulfonylurea (glipizide, glyburide, glimepiride), ask your care team about low sugar before adding walks after meals."
WALK_CAUTION = ("Because of your low readings, check with your care team before walking more. Walk after eating, "
                "not on an empty stomach, carry juice or glucose tablets, and stop to check your sugar if you feel shaky, sweaty or confused.")


THIS_WEEK = {"en": ("Today:", "This week:"), "es": ("Hoy:", "Esta semana:"), "ht": ("Jodi a:", "Semèn sa a:")}


def team_line(team_key, lang, urgency):
    """Care-team line; only level 2 lows (urgency "today") keep "Today:", every other flag says "This week:"."""
    t = TEXT[team_key][lang]
    old, new = THIS_WEEK[lang]
    return t.replace(old, new, 1) if urgency == "this-week" else t


def patient_view(recs, low_sugar, team_key, urgency=None):
    keys = []
    for rec in recs:
        k = ACTION_FOR_TITLE.get(rec["title"])
        if k == "walk" and low_sugar:
            k = "walk_low"
        if k:
            keys.append((rec["id"], k))
    keys = keys[:3]
    view = {}
    for lang in LANGS:
        view[lang] = {"careTeam": team_line(team_key, lang, urgency) if team_key else None,
                      "lowSugarWarning": TEXT["hypo"][lang] if low_sugar else None,
                      "actions": [{"id": rid, "text": TEXT[k][lang]} for rid, k in keys],
                      "help": TEXT["help"][lang],
                      "translation": "original" if lang == "en" else "machine-translated, needs review",
                      "reviewed": lang == "en"}
    view["note"] = "Spanish and Haitian Creole are machine-translated (AI-drafted) and not yet reviewed by a native speaker; review before clinical use."
    return view


def plain_summary(recs, low_sugar, team_key, urgency=None):
    """One or two plain English sentences: the care-team line, then up to three actions."""
    v = patient_view(recs, low_sugar, team_key, urgency)["en"]
    parts = ([v["careTeam"]] if v["careTeam"] else []) + [a["text"] for a in v["actions"]]
    return " ".join(parts)
