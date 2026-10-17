# C4 Glucose Guide: API contract

Base URL (deployed): `https://c4-glucose-guide.zay-259.workers.dev`
All `/api/c4/*` responses are JSON, `access-control-allow-origin: *`, cached 60 s. Examples below are real responses from that deployment (long arrays trimmed with `"…"`).
Data is synthetic (Crane Lab hackathon set): nobody in it is real.

## Survivor ids and status

- `survivorId` = `c4-<participant_id>`, e.g. `c4-N0632` (the synthetic participant id). Unknown ids return 404 `{"error":"unknown survivorId","survivorId":"…"}`.
- `status`: `"ok" | "watch" | "alert"`, from fixed CGM thresholds (international consensus targets) on the most recent sensor window (weeks 11-12, else weeks 1-2):
  - **alert**: time in range 70-180 < 70%, or > 5% of readings above 250, or > 4% below 70.
  - **watch**: time in range < 90%, or > 5% above 180, or average up 10+ mg/dL since weeks 1-2.
  - **ok**: otherwise.
- The roster is sorted alert → watch → ok, then by id.

## `GET /api/c4/survivors`

Roster tiles. 50 survivors (the 50 sensor wearers in the dataset's 100-person app sample).

```json
[
  {
    "survivorId": "c4-N0072",
    "displayName": "N0072 · 61F · ovarian",
    "status": "alert",
    "headline": "63% time in range (weeks 1-2); time in range below 70%",
    "updatedAt": "2024-08-01T19:15:00"
  },
  {
    "survivorId": "c4-N0085",
    "displayName": "N0085 · 67M · melanoma",
    "status": "alert",
    "headline": "64% time in range (weeks 11-12); time in range below 70%",
    "updatedAt": "2025-05-11T23:45:00"
  }
]
```

## `GET /api/c4/survivors/:id`

The survivor panel. Fields:

| field | type | notes |
|---|---|---|
| `survivorId`, `displayName`, `status`, `headline`, `summary`, `updatedAt` | string | `updatedAt` = last sensor reading |
| `statusReasons` | string[] | why the status was set |
| `profile` | object | age, sex, cancerType, yearsSinceTreatment, bloodSugarGroup, group (trial arm), workSchedule |
| `evidence[]` | `{id, label, value, unit, ci: [lo, hi] \| null, n, scope, method, caveat}` | `scope`: `person` (their data), `cohort` (program within-person estimate), `trial` (randomized arms), `real` (CGMacros real-data check), `rule` (thresholds) |
| `recommendations[]` | `{id, title, text, basis, evidenceIds[]}` | `basis`: `personal`, `cohort`, `trial`, `rule`. Every number in `text` is a value, CI bound or n of a listed evidence item, or a fixed wording constant (8 pm, 13 hours, 70-180, 95% CI…). The E2E check enforces this. |
| `charts` | object of plain arrays | see below |
| `dataNote` | string | synthetic-data notice |

Charts:
- `dailyProfile`: `hours[24]`, `series[{label: "weeks 1-2"|"weeks 11-12", values[24]}]` mean mg/dL per hour of day, `targetRange [70,180]`.
- `dinnerResponse`: `minutes [0,15,…,180]`; `you.early[]`, `you.late[]` (mean mg/dL above pre-meal; `null` if no such dinners), `you.nEarly`, `you.nLate`; `cohort.early[]`, `cohort.late[]`.
- `walkResponse`: `minutes`; `you.noWalk[]`, `you.walk[]`, `you.nNoWalk`, `you.nWalk`; `cohort.noWalk[]`, `cohort.walk[]`.
- `walkDose` (cohort): `bins[]`, `est[]`, `lo[]`, `hi[]`, `nMeals[]`: rise vs no walk, same person, by walk length.
- `overnightFast`: `hours[]`, `morningGlucose[]` paired per night (scatter).
- `checkups`: `metrics[]`, `week0[]`, `week12[]` (`null` if they left early).

Example (`c4-N0632`, trimmed):

```json
{
  "survivorId": "c4-N0632",
  "displayName": "N0632 · 76F · colorectal",
  "status": "alert",
  "statusReasons": [
    "more than 4% of readings below 70"
  ],
  "headline": "95% time in range (weeks 11-12); more than 4% of readings below 70",
  "summary": "76-year-old female, colorectal cancer, 1.3 years after treatment; normal at start; group: coaching; day job. Average sugar 85 mg/dL in weeks 11-12, 95% of readings in range. Median rise after dinner 46 mg/dL. Synthetic participant (made-up data).",
  "updatedAt": "2024-06-23T23:45:00",
  "profile": {
    "age": 76,
    "sex": "female",
    "cancerType": "colorectal",
    "yearsSinceTreatment": 1.3,
    "bloodSugarGroup": "normal",
    "group": "coaching",
    "workSchedule": "day job"
  },
  "evidence": [
    {
      "id": "tir",
      "label": "Time in range 70-180 mg/dL (weeks 11-12)",
      "value": 94.7,
      "unit": "%",
      "ci": null,
      "n": 1272,
      "scope": "person",
      "method": "Share of 15-min sensor readings between 70 and 180 mg/dL.",
      "caveat": null
    },
    {
      "id": "mean_glucose",
      "label": "Average blood sugar (weeks 11-12)",
      "value": 84.8,
      "unit": "mg/dL",
      "ci": null,
      "n": 1272,
      "scope": "person",
      "method": "Mean of 15-min sensor readings.",
      "caveat": null
    },
    {
      "id": "tar180",
      "label": "Readings above 180 mg/dL (weeks 11-12)",
      "value": 0,
      "unit": "%",
      "ci": null,
      "n": 1272,
      "scope": "person",
      "method": "Share of readings above 180 mg/dL.",
      "caveat": null
    },
    {
      "id": "walk_effect_own",
      "label": "Your sugar rise after meals with a walk vs without",
      "value": -18.4,
      "unit": "mg/dL",
      "ci": [
        -27.6,
        -10.1
      ],
      "n": 41,
      "scope": "person",
      "method": "Your meals only: regression of rise on walk, carbs and meal type; 95% CI from 300 bootstrap resamples of your meals.",
      "caveat": "You chose when to walk, so this is a pattern, not proof."
    },
    {
      "id": "cohort_late",
      "label": "Program-wide: rise after dinners at 8 pm or later vs earlier (same person)",
      "value": 9.1,
      "unit": "mg/dL",
      "ci": [
        7.1,
        11.2
      ],
      "n": 4711,
      "scope": "cohort",
      "method": "Within-person regression adjusted for carbs; night-shift workers excluded; cluster bootstrap 95% CI.",
      "caveat": "Association, not proof."
    },
    {
      "id": "trial_fast_hba1c",
      "label": "Randomized: HbA1c change, coaching + overnight fast vs coaching alone",
      "value": -0.03,
      "unit": "% points",
      "ci": [
        -0.06,
        -0.01
      ],
      "n": 1126,
      "scope": "trial",
      "method": "Difference in mean (week 12 - week 0) between randomized arms, completers; bootstrap 95% CI.",
      "caveat": "Fair comparison: groups were assigned at random."
    },
    {
      "id": "real_late_dinner",
      "label": "Real-data check (CGMacros, 45 real adults): dinner at 8 pm or later vs earlier",
      "value": 4.6,
      "unit": "mg/dL",
      "ci": [
        -1.5,
        11.4
      ],
      "n": 317,
      "scope": "real",
      "method": "Within-person, carbs adjusted; cluster bootstrap 95% CI.",
      "caveat": "Same direction as the program data, but the interval includes zero: not proven in real people yet."
    }
  ],
  "recommendations": [
    {
      "id": "talk_to_team",
      "title": "Tell your care team about low readings",
      "text": "5.3% of your readings were below 70 mg/dL (weeks 11-12); the usual target is under 4%. Low sugar can make you shaky or dizzy. Share this before your next visit.",
      "basis": "rule",
      "evidenceIds": [
        "tbr70",
        "status_rule"
      ]
    },
    {
      "id": "walk",
      "title": "Take a short walk after meals",
      "text": "Your sugar rose about 18.2 mg/dL less after meals when you walked. Aim for 10 to 20 minutes after your biggest meal.",
      "weightOnYourData": 0.72,
      "basis": "personal",
      "evidenceIds": [
        "walk_effect_own",
        "walk_effect_blend",
        "cohort_walk_10_19",
        "real_movement"
      ]
    },
    {
      "id": "late_dinner",
      "title": "Move dinner earlier",
      "text": "57.1% of your dinners started at 8 pm or later. Across the program, late dinners raised sugar about 9.1 mg/dL more than earlier ones in the same person, and a dinner at 8-9 pm raised it 15.9 mg/dL more than one before 6 pm. Try to start dinner before 8 pm.",
      "basis": "cohort",
      "evidenceIds": [
        "late_share",
        "cohort_late",
        "cohort_dinner_8_9pm",
        "real_late_dinner"
      ]
    },
    {
      "id": "overnight_fast",
      "title": "Hold off on longer fasts for now",
      "text": "5.3% of your readings were below 70 mg/dL, so check with your care team before stretching your overnight fast. Right now it averages 9.9 hours.",
      "basis": "rule",
      "evidenceIds": [
        "tbr70",
        "fast_avg",
        "status_rule"
      ]
    }
  ],
  "charts": {
    "dailyProfile": {
      "hours": [
        0,
        1,
        2,
        3,
        "…"
      ],
      "series": [
        {
          "label": "weeks 1-2",
          "values": [
            84.1,
            83.5,
            83.1,
            82.1,
            "…"
          ]
        },
        {
          "label": "weeks 11-12",
          "values": [
            81.3,
            80.5,
            79.4,
            78.9,
            "…"
          ]
        }
      ],
      "unit": "mg/dL",
      "targetRange": [
        70,
        180
      ]
    },
    "dinnerResponse": {
      "minutes": [
        0,
        15,
        30,
        45,
        "…"
      ],
      "unit": "mg/dL above pre-meal",
      "you": {
        "early": [
          0,
          8.5,
          32.5,
          45.8,
          "…"
        ],
        "late": [
          0,
          12.6,
          33.8,
          45.2,
          "…"
        ],
        "nEarly": 6,
        "nLate": 8
      },
      "cohort": "{early:[…], late:[…]}"
    },
    "walkResponse": "same shape as dinnerResponse with you.noWalk / you.walk / nNoWalk / nWalk and cohort.noWalk / cohort.walk",
    "walkDose": {
      "bins": [
        "none",
        "10-19 min",
        "20-29 min",
        "30+ min"
      ],
      "est": [
        0,
        -17.1,
        -20.1,
        -16.8
      ],
      "lo": [
        0,
        -18.8,
        -22.9,
        -19.1
      ],
      "hi": [
        0,
        -15.6,
        -17.2,
        -14.5
      ],
      "nMeals": [
        16679,
        866,
        311,
        362
      ],
      "unit": "mg/dL vs no walk (same person)"
    },
    "overnightFast": {
      "hours": [
        10.4,
        11.6,
        8.7,
        9.6,
        9.3,
        "…"
      ],
      "morningGlucose": [
        86,
        88,
        94,
        80,
        88,
        "…"
      ],
      "unit": "mg/dL"
    },
    "checkups": {
      "metrics": [
        "hba1c_pct",
        "fasting_glucose_mg_dl",
        "weight_kg"
      ],
      "week0": [
        5.1,
        83,
        75.9
      ],
      "week12": [
        5,
        84,
        77.1
      ]
    }
  },
  "dataNote": "Synthetic data from the Crane Lab hackathon set. Nobody in it is real."
}
```

## `GET /api/c4/survivors/:id/advice`

Optional. Workers AI (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`) rewrites the recommendations in plain language, about 3-5 s. Each rewrite is checked: if it uses any number that is not in the template text or its evidence, it is dropped and the template is returned (`source: "template"`, `rejected: {text, untracedNumbers}`).

```json
{
  "survivorId": "c4-N0632",
  "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "ms": 4043,
  "error": null,
  "rawPreview": null,
  "items": [
    {
      "id": "talk_to_team",
      "title": "Tell your care team about low readings",
      "text": "You had low sugar readings 5.3% of the time, which can make you feel shaky or dizzy, so share this with your team before your next visit. Your target is to have low sugar readings under 4% of the time.",
      "source": "ai",
      "evidenceIds": [
        "tbr70",
        "status_rule"
      ],
      "rejected": null
    },
    {
      "id": "walk",
      "title": "Take a short walk after meals",
      "text": "Walking after meals helped lower your sugar by about 18.2 mg/dL, which is a great start. Try to walk for 10 to 20 minutes after your biggest meal to see how it helps.",
      "source": "ai",
      "evidenceIds": [
        "walk_effect_own",
        "walk_effect_blend",
        "cohort_walk_10_19",
        "real_movement"
      ],
      "rejected": null
    }
  ]
}
```

## `GET /api/c4/cohort`

Cohort findings for a program-level view. Top-level keys:

```json
{
  "source": "str",
  "effects": [
    "lateDinner",
    "lateDinnerIauc",
    "walk",
    "walkIauc",
    "carbsPer10g",
    "fastMorningPerHour",
    "fast13Morning"
  ],
  "effectsByBloodSugarGroup": [
    "normal",
    "prediabetes",
    "type 2 diabetes"
  ],
  "walkDose": "list",
  "dinnerHour": "list",
  "curves": [
    "minutes",
    "earlyDinner",
    "lateDinner",
    "noWalk",
    "walk"
  ],
  "trial": [
    "arms",
    "contrasts"
  ],
  "statusCounts": [
    "ok",
    "watch",
    "alert"
  ],
  "realWorld": [
    "cgmacros",
    "nhanesSurvivors",
    "methods"
  ],
  "methods": "str"
}
```

- `effects.walk`, `effects.lateDinner`, `effects.carbsPer10g`, `effects.fastMorningPerHour`, … : `{est, lo, hi, people, rows}` within-person (person fixed effects) estimates with cluster-bootstrap 95% CIs.
- `walkDose[]`, `dinnerHour[]`: `{bin, est, lo, hi, n_meals}` for dose-response charts.
- `trial.contrasts.<outcome>["coaching + overnight fast vs coaching"]`: randomized difference in week 12 minus week 0 change.
- `realWorld.cgmacros`, `realWorld.nhanesSurvivors`: the same questions in real data.

```json
{
  "walk": {
    "est": -17.62,
    "lo": -18.98,
    "hi": -16.26,
    "people": 500,
    "rows": 18218
  },
  "lateDinner": {
    "est": 9.08,
    "lo": 7.05,
    "hi": 11.23,
    "people": 471,
    "rows": 4711
  },
  "walkDose[1]": {
    "bin": "10-19 min",
    "est": -17.13,
    "lo": -18.81,
    "hi": -15.63,
    "people": 500,
    "rows": 18218,
    "n_meals": 866
  },
  "trial hba1c": {
    "est": -0.03,
    "lo": -0.06,
    "hi": -0.01,
    "n": 1126
  },
  "realWorld.cgmacros.lateDinner": {
    "est": 4.62,
    "lo": -1.517,
    "hi": 11.37,
    "people": 41,
    "rows": 317
  }
}
```

## `GET /`

Standalone demo page (`public/index.html`): roster, survivor panel with charts and evidence, and a **Cohort findings** tab (`/#cohort`).
