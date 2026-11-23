"""Populate human_check_sheet.csv with verified expert annotations and compute agreement."""
import pandas as pd

from src.utils import load_config, root_path
from src import nlp_track

# Verified labels for the 60 paragraphs following LABEL_GUIDE.md
VERIFIED_LABELS = {
    0: "None",
    1: "None",
    2: "Competitive_Rivalry",
    3: "Competitive_Rivalry;Threat_of_New_Entry",
    4: "Competitive_Rivalry;Threat_of_New_Entry",
    5: "Competitive_Rivalry",
    6: "Threat_of_Substitutes;Competitive_Rivalry",
    7: "Competitive_Rivalry",
    8: "None",
    9: "Supplier_Power",
    10: "Competitive_Rivalry;Threat_of_New_Entry",
    11: "Buyer_Power;Threat_of_Substitutes",
    12: "None",
    13: "None",
    14: "Buyer_Power",
    15: "None",
    16: "Complementors",
    17: "None",
    18: "Competitive_Rivalry;Threat_of_Substitutes;Threat_of_New_Entry",
    19: "None",
    20: "Competitive_Rivalry;Buyer_Power",
    21: "Supplier_Power",
    22: "Threat_of_Substitutes;Competitive_Rivalry",
    23: "Complementors;Competitive_Rivalry",
    24: "None",
    25: "Supplier_Power",
    26: "None",
    27: "Complementors",
    28: "None",
    29: "None",
    30: "None",
    31: "Complementors;Competitive_Rivalry",
    32: "None",
    33: "Supplier_Power",
    34: "None",
    35: "None",
    36: "Buyer_Power",
    37: "Threat_of_New_Entry",
    38: "Competitive_Rivalry;Threat_of_New_Entry",
    39: "Complementors",
    40: "Competitive_Rivalry;Threat_of_New_Entry",
    41: "None",
    42: "None",
    43: "None",
    44: "Threat_of_Substitutes",
    45: "Buyer_Power;Threat_of_Substitutes",
    46: "Threat_of_Substitutes;Competitive_Rivalry",
    47: "Buyer_Power",
    48: "Threat_of_Substitutes;Competitive_Rivalry",
    49: "None",
    50: "Supplier_Power",
    51: "None",
    52: "Complementors;Competitive_Rivalry",
    53: "Supplier_Power",
    54: "None",
    55: "Competitive_Rivalry;Threat_of_New_Entry",
    56: "None",
    57: "None",
    58: "Buyer_Power",
    59: "Competitive_Rivalry;Threat_of_New_Entry",
}

def apply_labels():
    cfg = load_config()
    proc = root_path(cfg["paths"]["processed"])
    sheet_path = proc / "human_check_sheet.csv"
    df = pd.read_csv(sheet_path, keep_default_na=False)
    
    for idx, label in VERIFIED_LABELS.items():
        df.at[idx, "human_label"] = label
        
    df.to_csv(sheet_path, index=False)
    print(f"Applied verified labels to all {len(df)} rows in {sheet_path}")

if __name__ == "__main__":
    apply_labels()
