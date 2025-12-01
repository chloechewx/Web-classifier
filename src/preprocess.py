import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)

def general_clean(df: pd.DataFrame) -> pd.DataFrame:
     
    if df is None:
        raise ValueError("general_clean(df) requires a DataFrame, but none was received.")

    df = df.copy()

    # Drop index column 'Unnamed:0'
    df = df.drop(columns=["Unnamed: 0", "Unnamed:0"], errors="ignore")
    logging.info("Dropped Unnamed index columns if present.")

    # Add Missing flag column for `Null` values in `LineOfCode` to preserve signal.
    if "LineOfCode" in df.columns:
        df["LineOfCode_missing"] = df["LineOfCode"].isna().astype(int)
        logging.info("Created LineOfCode_missing flag column.")
    else:
        logging.warning("LineOfCode column not found in DataFrame.")

    # Impute median into the missing values in `LineOfCode` to represent typical value, prevent introducing fake patterns.
    if "LineOfCode" in df.columns:
        median_loc = df["LineOfCode"].median()
        df["LineOfCode"] = df["LineOfCode"].fillna(median_loc)
        logging.info(f"Filled missing LineOfCode values with median: {median_loc}")

        # Type correction in `LineofCode` (`Float` to `Int`)
        df["LineOfCode"] = df["LineOfCode"].round().astype(int)
        logging.info("Converted LineOfCode from Float to Int")

    # Unifying categorical labels in `Industry` ("Ecommerce" and "Ecommerce ").
    if "Industry" in df.columns:
        df["Industry"] = df["Industry"].astype(str).str.strip()
        logging.info("Stripped whitespace from Industry labels.")
    else:
        logging.warning("Industry column not found in DataFrame.")

    # `NoofImage` negative values will be clipped to 0
    if "NoOfImage" in df.columns:
        df["NoOfImage"] = df["NoOfImage"].clip(lower=0)
        logging.info("Clipped negative values in NoOfImage to 0.")
    else:
        logging.warning("NoOfImage column not found in DataFrame.")
        
    return df



# cleaned_df = general_clean(df)
# cleaned_df.to_csv("cleaned_df.csv", index=False)
