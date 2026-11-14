import pandas as pd

def filter_large_csv_by_stay_ids(input_file, output_file, key_col, chunksize=100_000):
    # Eye, verbal, motor
    valid_ids = [220739, 223900, 223901]

    reader = pd.read_csv(input_file, chunksize=chunksize)
    first_chunk = True

    total_rows_kept = 0
    for i, chunk in enumerate(reader):
        filtered_chunk = chunk[chunk[key_col].isin(valid_ids)]
        total_rows_kept += len(filtered_chunk)

        if first_chunk:
            filtered_chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)

        print(f"✔ 已处理第 {i+1} 块，保留 {len(filtered_chunk)} 行")

    print(f"🎉 筛选完成，最终保留 {total_rows_kept} 行，保存至 {output_file}")

def process_columns(data_file, output_file):

    df = pd.read_csv(data_file)

    df = df[["subject_id", "charttime", "itemid", "value", "valuenum"]]

    df.rename(columns={'itemid': 'item'}, inplace=True)

    mapping = {220739: 'eye', 223900: 'verbal', 223901: 'motor'}
    df["item"] = df["item"].map(mapping).fillna(df["item"])

    df.sort_values(by=['subject_id', 'charttime'], inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def calculate_gcs(input_path, output_path):
    """Calculate total GCS using original component scores within four hours."""

    df = pd.read_csv(
        input_path,
        dtype={"subject_id": "string"},
    )

    required = {"subject_id", "charttime", "item", "value", "valuenum"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df = df.loc[df["item"].isin(["eye", "motor", "verbal"])].copy()
    df["charttime"] = pd.to_datetime(
        df["charttime"], errors="raise"
    ).astype("datetime64[ns]")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="raise")

    keys = ["subject_id", "charttime"]

    if df[keys].isna().any().any():
        raise ValueError("subject_id and charttime must not be missing.")

    # Identify timestamps with an explicit intubated verbal response.
    ett_times = (
        df.loc[
            df["item"].eq("verbal")
            & df["value"].eq("No Response-ETT"),
            keys,
        ]
        .drop_duplicates()
    )

    # Exclude every component at ETT timestamps from all filling sources.
    ett_index = pd.MultiIndex.from_frame(ett_times)
    is_ett = pd.MultiIndex.from_frame(df[keys]).isin(ett_index)
    normal = df.loc[~is_ett].copy()

    # Keep each non-ETT timestamp as a candidate for total GCS.
    candidates = (
        normal[keys]
        .drop_duplicates()
        .sort_values(["charttime", "subject_id"])
        .reset_index(drop=True)
    )

    # Reject conflicting scores instead of choosing one arbitrarily.
    score_counts = (
        normal.groupby(keys + ["item"])["valuenum"].nunique()
    )
    conflicts = score_counts.loc[score_counts.gt(1)]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting scores for the same subject, timestamp, and component:\n"
            f"{conflicts.head(10).to_string()}"
        )

    # Collapse identical duplicates and retain only original numeric scores.
    sources = (
        normal.dropna(subset=["valuenum"])
        .groupby(keys + ["item"], as_index=False)["valuenum"]
        .max()
    )

    for component in ("eye", "motor", "verbal"):
        source = sources.loc[
            sources["item"].eq(component),
            keys + ["valuenum"],
        ].rename(columns={"valuenum": component})

        if candidates.empty or source.empty:
            candidates[component] = float("nan")
            continue

        # Prefer an exact match; otherwise use the latest original score
        # from the same subject within the preceding four hours.
        candidates = pd.merge_asof(
            candidates.sort_values("charttime", kind="stable"),
            source.sort_values("charttime", kind="stable"),
            on="charttime",
            by="subject_id",
            direction="backward",
            tolerance=pd.Timedelta(hours=4),
            allow_exact_matches=True,
        )

    # Discard timestamps that still lack any of the three components.
    complete = candidates.dropna(subset=["eye", "motor", "verbal"]).copy()
    complete["value"] = complete[["eye", "motor", "verbal"]].sum(axis=1)

    # ETT timestamps always receive a total score of 15.
    ett_result = ett_times.assign(value=15)

    result = (
        pd.concat(
            [complete[keys + ["value"]], ett_result],
            ignore_index=True,
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )

    result.to_csv(output_path, index=False)

    return result

if __name__ == "__main__":

    #filter_large_csv_by_stay_ids('../../icu/chartevents.csv', './gcs_raw.csv', 'itemid')
    #process_columns('./gcs_raw.csv', './gcs_raw.csv')
    calculate_gcs("./gcs_raw.csv", "./gcs.csv")
    