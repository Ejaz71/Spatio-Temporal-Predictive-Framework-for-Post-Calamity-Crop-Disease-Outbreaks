"""
Builds the upazila-level (72-row) rice blast event table from the real Mahmud et al.
(2021) upazila x year raw survey data, applying the SAME outbreak threshold rule as
the district-level events.csv (proposal Section 6.3):
    label = 1 if leaf_blast_incidence_pct >= 15 OR neck_blast_incidence_pct >= 8

Input:  data/raw_literature/sajbr2021/upazila_x_year_raw.csv (72 real survey rows)
        config/upazila_gazetteer.json (24 real upazila coordinates, looked up
        individually from each upazila's Wikipedia infobox — see that file's
        "sources" field for the exact fetch date)
Output: data/raw_literature/sajbr2021/upazila_events.csv — same column schema as
        events.csv, plus `upazila`/`lat`/`lon` columns so real_feature_pipeline.py
        can query NASA POWER / Sentinel-1 / Landsat per upazila centroid instead of
        per district centroid.

Run from the repo root: python3 data/build_upazila_events.py
"""
import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAZETTEER_JSON = os.path.join(BASE, "config", "upazila_gazetteer.json")
RAW_CSV = os.path.join(BASE, "data", "raw_literature", "sajbr2021", "upazila_x_year_raw.csv")
OUT_CSV = os.path.join(BASE, "data", "raw_literature", "sajbr2021", "upazila_events.csv")

FIELDNAMES = [
    "event_id", "disease", "district", "upazila", "lat", "lon", "season_label", "year",
    "window_start", "window_end", "label", "leaf_blast_incidence_pct", "neck_blast_incidence_pct",
    "leaf_blast_severity_pct", "neck_blast_severity_pct", "pct_fields_infected",
    "avg_yield_loss_pct", "affected_area_ha", "note", "source",
]


def main():
    with open(GAZETTEER_JSON) as f:
        gaz = json.load(f)["upazilas"]

    rows_out = []
    with open(RAW_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            district = r["district"]
            upazila = r["upazila"]
            year = int(r["year"])
            lbi = float(r["leaf_blast_incidence_pct"])
            nbi = float(r["neck_blast_incidence_pct"])
            lbs = float(r["leaf_blast_severity_pct"])
            nbs = float(r["neck_blast_severity_pct"])

            key = f"{district}|{upazila}"
            if key not in gaz:
                raise KeyError(f"No gazetteer entry for {key}")
            lat, lon = gaz[key]["lat"], gaz[key]["lon"]

            label = 1 if (lbi >= 15 or nbi >= 8) else 0
            reasons = []
            if lbi >= 15:
                reasons.append(f"LBI={lbi:g}% >= 15%")
            if nbi >= 8:
                reasons.append(f"NBI={nbi:g}% >= 8%")
            note = "; ".join(reasons) if reasons else f"LBI={lbi:g}% < 15% and NBI={nbi:g}% < 8%"

            rows_out.append({
                "event_id": f"RB_{district}_{upazila}_{year}",
                "disease": "rice_blast",
                "district": district,
                "upazila": upazila,
                "lat": lat,
                "lon": lon,
                "season_label": f"Boro {year}",
                "year": year,
                "window_start": f"{year-1}-12-01",
                "window_end": f"{year}-03-15",
                "label": label,
                "leaf_blast_incidence_pct": lbi,
                "neck_blast_incidence_pct": nbi,
                "leaf_blast_severity_pct": lbs,
                "neck_blast_severity_pct": nbs,
                "pct_fields_infected": "",
                "avg_yield_loss_pct": "",
                "affected_area_ha": "",
                "note": note,
                "source": r["source"],
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    n_pos = sum(r["label"] for r in rows_out)
    print(f"Wrote {len(rows_out)} upazila-level events to {OUT_CSV}")
    print(f"Positive (outbreak) events: {n_pos} / {len(rows_out)}")
    print(f"Unique upazilas: {len(set((r['district'], r['upazila']) for r in rows_out))}")


if __name__ == "__main__":
    main()
