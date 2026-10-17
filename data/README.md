# Data

No data is committed. Everything comes from the eMerge hackathon Challenge 4 package, `Challenge4_Dose_and_Timing.zip` (sponsor data; the README inside calls it "Challenge 2: Dose and Timing"). Ask the organizers for it; it is not redistributed here.

Zip sha256: `945a6c07fd95104eaa395430a5ffbb1c2f37db7c72d675b1b2ac5911080f68a4` (318 MB, including meal photos that nothing here uses).

## Where it goes

Unzip it anywhere outside the repo and pass the folder to `scripts/reproduce.sh`. The folder must hold these files (the meal photos and `CGMacros/original_data_dictionaries/` are optional):

```
Challenge4_Dose_and_Timing/
  README.md
  nhanes_us_adults.csv
  synthetic_data/synthetic_participants.csv
  synthetic_data/synthetic_checkups_week0_week12.csv
  synthetic_data/synthetic_daily.csv
  synthetic_data/synthetic_meals.csv
  synthetic_data/synthetic_glucose_every_15_min.csv
  synthetic_data/synthetic_100_people.json
  synthetic_data/README.txt
  CGMacros/cgmacros_participants.csv
  CGMacros/cgmacros_meals.csv
  CGMacros/cgmacros_minute_by_minute.csv
  CGMacros/cgmacros_gut_health.csv
  CGMacros/cgmacros_gut_microbes.csv
```

`scripts/reproduce.sh` checks each file against [`SHA256SUMS`](SHA256SUMS) first and stops if any differs. To check by hand: `cd Challenge4_Dose_and_Timing && sha256sum -c <repo>/data/SHA256SUMS` (macOS: `shasum -a 256 -c`).

## What each part is, and its public source

| Part | What it is | Public source, version, access |
|---|---|---|
| `synthetic_data/` | Made-up 12-week program: 2,000 people, 305,154 meals, 15-minute glucose for 500 sensor wearers. The headline effects come from here. | Made by the organizers (Crane Lab) for the hackathon; not public. Only in the package. |
| `CGMacros/` | 45 real adults, 10 days of Libre CGM, Fitbit and logged meals. The real-data check. | Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. *CGMacros* v1.0.0, PhysioNet 2025, https://physionet.org/content/cgmacros/1.0.0/. Open access for research (see the license on that page). The package ships the organizers' tidied extract, which these scripts expect. |
| `nhanes_us_adults.csv` | 11,329 US adults, 808 cancer survivors: meal timing, accelerometer movement, HbA1c, survey weights. | CDC, National Health and Nutrition Examination Survey 2011-2014 with the 2019 linked mortality files, https://wwwn.cdc.gov/nchs/nhanes/. Public domain, no access requirement; use it for statistics only and never try to identify anyone. The package ships the organizers' merged extract. |
