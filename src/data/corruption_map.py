"""
Corruption registry for HuGaDB gyroscope data.

The corruption is a 10x amplification of gyroscope signals, documented by the
dataset authors. This module provides a lookup to determine which gyroscope
channel groups (RF, RS, RT, LF, LS, LT) are corrupted for each file.

Usage:
    from src.data.corruption_map import get_corrupted_gyro_groups
    corrupted = get_corrupted_gyro_groups("HuGaDB_v1_walking_04_00.txt")
    # Returns: ['RF', 'RS', 'RT', 'LF', 'LS', 'LT']  (all corrupted)
"""


def _make_entry(rf, rs, rt, lf, ls, lt):
    """Helper: returns list of corrupted location codes from boolean flags."""
    groups = []
    if rf: groups.append("RF")
    if rs: groups.append("RS")
    if rt: groups.append("RT")
    if lf: groups.append("LF")
    if ls: groups.append("LS")
    if lt: groups.append("LT")
    return groups

# Shorthand corruption patterns
ALL   = _make_entry(True,  True,  True,  True,  True,  True)
RIGHT = _make_entry(True,  True,  True,  False, False, False)  # RF, RS, RT
LEFT  = _make_entry(False, False, False, True,  True,  True)   # LF, LS, LT

# Right foot + right shank only
RF_RS     = _make_entry(True,  True,  False, False, False, False)
RF_RS_RT  = RIGHT

# PATTERN: various_03 → right side corrupted
_various_03 = RIGHT
# PATTERN: various_04 → ALL corrupted
_various_04 = ALL
# PATTERN: various_05_00 → RF+RS corrupted, rest varies
_various_05_00 = _make_entry(True, True, False, False, False, False)
_various_05_rest = _make_entry(True, True, True, True, False, False)
# PATTERN: various_07 → RF + RS corrupted
_various_07 = _make_entry(True, True, False, False, False, False)
# PATTERN: various_09_00 to 08 → ALL
_various_09_all = ALL
# PATTERN: various_09_13 to 20 → LEFT side
_various_09_left = LEFT
# PATTERN: various_11 → LF only
_various_11 = _make_entry(False, False, False, True, False, False)
# PATTERN: various_12, 13, 14, 15, 16, 17 → ALL
# PATTERN: various_18 → all except RT
_various_18 = _make_entry(True, True, True, True, False, True)


def _build_corruption_map():
    """Build the complete corruption map from the known corruption table."""
    cmap = {}

    # ---- VARIOUS files ----
    # various_03: 00-24, right side corrupted
    for i in range(25):
        cmap[f"HuGaDB_v1_various_03_{i:02d}.txt"] = RIGHT

    # various_04: 00-19, ALL corrupted
    for i in range(20):
        cmap[f"HuGaDB_v1_various_04_{i:02d}.txt"] = ALL

    # various_05: 00 has RF+RS; 01-20 have RF+RS+RT+LF (not LS, LT)
    cmap["HuGaDB_v1_various_05_00.txt"] = _make_entry(True, True, False, False, False, False)
    for i in range(1, 21):
        cmap[f"HuGaDB_v1_various_05_{i:02d}.txt"] = _make_entry(True, True, True, True, False, False)

    # various_07: 00-24, RF+RS corrupted
    for i in range(25):
        cmap[f"HuGaDB_v1_various_07_{i:02d}.txt"] = _make_entry(True, True, False, False, False, False)

    # various_09: 00-08 ALL, 13-20 LEFT
    for i in range(9):
        cmap[f"HuGaDB_v1_various_09_{i:02d}.txt"] = ALL
    for i in range(13, 21):
        cmap[f"HuGaDB_v1_various_09_{i:02d}.txt"] = LEFT

    # various_11: 04-20, LF only
    for i in range(4, 21):
        cmap[f"HuGaDB_v1_various_11_{i:02d}.txt"] = _make_entry(False, False, False, True, False, False)

    # various_12: 00-18, ALL
    for i in range(19):
        cmap[f"HuGaDB_v1_various_12_{i:02d}.txt"] = ALL
    # Fix non-sequential IDs (10,11 appear after 09 in user's list)
    cmap["HuGaDB_v1_various_12_10.txt"] = ALL
    cmap["HuGaDB_v1_various_12_11.txt"] = ALL

    # various_13: 07-24, ALL
    for i in range(7, 25):
        cmap[f"HuGaDB_v1_various_13_{i:02d}.txt"] = ALL

    # various_14: 00-21, ALL
    for i in range(22):
        cmap[f"HuGaDB_v1_various_14_{i:02d}.txt"] = ALL

    # various_15: 00-11, ALL
    for i in range(12):
        cmap[f"HuGaDB_v1_various_15_{i:02d}.txt"] = ALL

    # various_16: 00-19, ALL
    for i in range(20):
        cmap[f"HuGaDB_v1_various_16_{i:02d}.txt"] = ALL

    # various_17: 00-23, ALL (03 missing in user's list, include anyway)
    for i in range(24):
        cmap[f"HuGaDB_v1_various_17_{i:02d}.txt"] = ALL

    # various_18: 00-16, all except RT
    for i in range(17):
        cmap[f"HuGaDB_v1_various_18_{i:02d}.txt"] = _make_entry(True, True, True, True, False, True)

    # ---- RUNNING files ----
    for i in range(2):
        cmap[f"HuGaDB_v1_running_03_{i:02d}.txt"] = RIGHT
    for i in range(3):
        cmap[f"HuGaDB_v1_running_07_{i:02d}.txt"] = _make_entry(True, True, False, False, False, False)
    for i in range(3):
        cmap[f"HuGaDB_v1_running_09_{i:02d}.txt"] = LEFT

    # ---- SITTING files ----
    # sitting_03
    cmap["HuGaDB_v1_sitting_03_00.txt"] = _make_entry(True, False, True, False, False, False)
    for i in range(1, 4):
        cmap[f"HuGaDB_v1_sitting_03_{i:02d}.txt"] = RIGHT
    # sitting_04
    for i in range(3):
        cmap[f"HuGaDB_v1_sitting_04_{i:02d}.txt"] = ALL
    # sitting_05
    cmap["HuGaDB_v1_sitting_05_00.txt"] = ALL
    cmap["HuGaDB_v1_sitting_05_01.txt"] = _make_entry(False, False, False, True, False, False)
    # sitting_06
    cmap["HuGaDB_v1_sitting_06_04.txt"] = _make_entry(True, True, False, False, False, True)
    # sitting_07
    cmap["HuGaDB_v1_sitting_07_01.txt"] = _make_entry(True, True, False, False, False, False)
    cmap["HuGaDB_v1_sitting_07_02.txt"] = ALL
    cmap["HuGaDB_v1_sitting_07_03.txt"] = _make_entry(True, True, False, False, False, False)
    cmap["HuGaDB_v1_sitting_07_04.txt"] = _make_entry(True, True, False, False, False, False)
    # sitting_08
    cmap["HuGaDB_v1_sitting_08_00.txt"] = ALL
    # sitting_09
    for i in range(4):
        cmap[f"HuGaDB_v1_sitting_09_{i:02d}.txt"] = ALL
    # sitting_10
    cmap["HuGaDB_v1_sitting_10_00.txt"] = ALL
    cmap["HuGaDB_v1_sitting_10_04.txt"] = ALL
    # sitting_11
    cmap["HuGaDB_v1_sitting_11_00.txt"] = _make_entry(False, False, False, True, False, False)
    cmap["HuGaDB_v1_sitting_11_01.txt"] = _make_entry(False, False, False, True, True, True)
    cmap["HuGaDB_v1_sitting_11_02.txt"] = _make_entry(False, False, False, True, False, False)
    cmap["HuGaDB_v1_sitting_11_03.txt"] = _make_entry(False, True, False, False, True, True)
    # sitting_12
    for i in range(4):
        cmap[f"HuGaDB_v1_sitting_12_{i:02d}.txt"] = ALL
    # sitting_13
    for i in range(7):
        cmap[f"HuGaDB_v1_sitting_13_{i:02d}.txt"] = ALL
    # sitting_14
    for i in range(4):
        cmap[f"HuGaDB_v1_sitting_14_{i:02d}.txt"] = ALL
    # sitting_15
    cmap["HuGaDB_v1_sitting_15_00.txt"] = _make_entry(True, False, True, True, True, True)
    cmap["HuGaDB_v1_sitting_15_01.txt"] = ALL
    cmap["HuGaDB_v1_sitting_15_02.txt"] = _make_entry(True, False, True, True, True, True)
    cmap["HuGaDB_v1_sitting_15_03.txt"] = ALL
    # sitting_16
    for i in range(6):
        cmap[f"HuGaDB_v1_sitting_16_{i:02d}.txt"] = ALL
    # sitting_17
    for i in range(7):
        cmap[f"HuGaDB_v1_sitting_17_{i:02d}.txt"] = ALL
    # sitting_18
    cmap["HuGaDB_v1_sitting_18_00.txt"] = ALL
    cmap["HuGaDB_v1_sitting_18_01.txt"] = ALL
    cmap["HuGaDB_v1_sitting_18_02.txt"] = _make_entry(True, True, True, True, False, True)
    cmap["HuGaDB_v1_sitting_18_03.txt"] = _make_entry(True, True, True, True, False, True)

    # ---- SITTING IN CAR ----
    for i in range(15):
        fname = f"HuGaDB_v1_sitting_in_car_01_{i:02d}.txt"
        cmap[fname] = ALL
    # Note: 01_07 was not listed but all others 00-14 are; skip 07 if not in list
    cmap.pop("HuGaDB_v1_sitting_in_car_01_07.txt", None)

    # ---- STANDING files ----
    cmap["HuGaDB_v1_standing_01_00.txt"] = _make_entry(False, True, True, False, False, True)
    cmap["HuGaDB_v1_standing_01_03.txt"] = _make_entry(False, False, False, True, False, False)
    cmap["HuGaDB_v1_standing_03_00.txt"] = _make_entry(True, True, True, False, False, True)
    for i in range(4):
        cmap[f"HuGaDB_v1_standing_04_{i:02d}.txt"] = ALL
    cmap["HuGaDB_v1_standing_05_00.txt"] = ALL
    cmap["HuGaDB_v1_standing_05_01.txt"] = _make_entry(True, True, False, False, False, False)
    cmap["HuGaDB_v1_standing_06_00.txt"] = _make_entry(False, False, False, False, True, True)
    cmap["HuGaDB_v1_standing_07_00.txt"] = _make_entry(True, True, False, False, True, False)
    cmap["HuGaDB_v1_standing_07_01.txt"] = _make_entry(False, False, False, False, True, True)
    cmap["HuGaDB_v1_standing_07_03.txt"] = ALL
    cmap["HuGaDB_v1_standing_08_00.txt"] = _make_entry(False, False, False, False, True, True)
    cmap["HuGaDB_v1_standing_08_01.txt"] = _make_entry(False, False, False, False, True, True)
    cmap["HuGaDB_v1_standing_08_02.txt"] = _make_entry(False, True, True, False, True, True)
    cmap["HuGaDB_v1_standing_08_03.txt"] = _make_entry(True, False, True, False, True, True)
    for i in range(5):
        cmap[f"HuGaDB_v1_standing_09_{i:02d}.txt"] = LEFT
    cmap["HuGaDB_v1_standing_09_04.txt"] = _make_entry(False, True, True, True, True, True)
    cmap["HuGaDB_v1_standing_10_01.txt"] = _make_entry(False, True, True, False, True, False)
    cmap["HuGaDB_v1_standing_10_03.txt"] = _make_entry(False, True, False, False, False, False)
    cmap["HuGaDB_v1_standing_11_00.txt"] = _make_entry(False, False, False, True, True, True)
    cmap["HuGaDB_v1_standing_11_01.txt"] = _make_entry(False, True, True, True, True, True)
    cmap["HuGaDB_v1_standing_11_02.txt"] = _make_entry(False, True, True, True, True, True)
    cmap["HuGaDB_v1_standing_11_03.txt"] = _make_entry(False, False, False, True, True, True)
    cmap["HuGaDB_v1_standing_11_04.txt"] = _make_entry(False, False, False, True, False, True)
    for i in range(5):
        cmap[f"HuGaDB_v1_standing_12_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_standing_13_{i:02d}.txt"] = ALL
    for i in range(5):
        cmap[f"HuGaDB_v1_standing_15_{i:02d}.txt"] = ALL
    for i in range(5):
        cmap[f"HuGaDB_v1_standing_16_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_standing_17_{i:02d}.txt"] = ALL
    for i in range(5):
        cmap[f"HuGaDB_v1_standing_18_{i:02d}.txt"] = ALL

    # ---- WALKING files ----
    for i in range(2):
        cmap[f"HuGaDB_v1_walking_03_{i:02d}.txt"] = RIGHT
    for i in range(6):
        cmap[f"HuGaDB_v1_walking_04_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_walking_05_{i:02d}.txt"] = _make_entry(True, True, True, False, False, False)
    cmap["HuGaDB_v1_walking_06_03.txt"] = _make_entry(False, False, True, False, False, False)
    cmap["HuGaDB_v1_walking_06_05.txt"] = _make_entry(False, False, True, False, False, False)
    for i in range(5):
        cmap[f"HuGaDB_v1_walking_07_{i:02d}.txt"] = _make_entry(True, True, False, False, False, False)
    for i in range(4):
        cmap[f"HuGaDB_v1_walking_09_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_walking_11_{i:02d}.txt"] = _make_entry(False, False, False, True, False, False)
    for i in range(7):
        cmap[f"HuGaDB_v1_walking_12_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_walking_13_{i:02d}.txt"] = ALL
    for i in range(5):
        cmap[f"HuGaDB_v1_walking_14_{i:02d}.txt"] = ALL
    for i in range(11):
        cmap[f"HuGaDB_v1_walking_15_{i:02d}.txt"] = ALL
    for i in range(5):
        cmap[f"HuGaDB_v1_walking_16_{i:02d}.txt"] = ALL
    for i in range(7):
        cmap[f"HuGaDB_v1_walking_17_{i:02d}.txt"] = ALL
    for i in range(4):
        cmap[f"HuGaDB_v1_walking_18_{i:02d}.txt"] = _make_entry(True, True, True, True, False, True)

    return cmap


# Build the map once at import time
CORRUPTION_MAP = _build_corruption_map()


def get_corrupted_gyro_groups(filename):
    """
    Get list of corrupted gyroscope location groups for a given filename.

    Args:
        filename: Just the filename (e.g., 'HuGaDB_v1_walking_04_00.txt')
                  or a full path — only the basename is used.

    Returns:
        List of corrupted location codes, e.g., ['RF', 'RS', 'RT', 'LF', 'LS', 'LT']
        Empty list if the file has no corruption.
    """
    import os
    basename = os.path.basename(filename)
    return CORRUPTION_MAP.get(basename, [])


def is_corrupted(filename):
    """Return True if the file has any corrupted gyroscope channels."""
    return len(get_corrupted_gyro_groups(filename)) > 0


def get_corrupted_gyro_columns(filename):
    """
    Get the specific gyroscope column names that are corrupted for a file.

    Returns:
        List of column names, e.g., ['gyro_rf_x', 'gyro_rf_y', 'gyro_rf_z', ...]
    """
    from src.config import GYRO_GROUPS
    groups = get_corrupted_gyro_groups(filename)
    columns = []
    for group in groups:
        columns.extend(GYRO_GROUPS[group])
    return columns
