import numpy as np

# FILE PATH
RAW_FILE = ""
ENV_FILE = ""
TEMP_DF_PATH = ""

# DATA SETTINGS
SEP = ';'
DECIMAL = '.'
TARGET_COL = "onset_age"
EXPOSURE = ""
COVARIATES = ["sex"]

# MATCHING
MATCH_K = 3
MIN_TREATED = 5
MIN_SAMPLE_SIZE = 10

# PERMUTATION
N_PERM = 500
RANDOM_STATE = 42
MIN_OBS_COEF = 2

# SCALING
STANDARDIZE = True

# PARALLEL
MAX_WORKERS = 16

# DATABASE
DB_USER = ''
DB_PASSWORD = ''
DB_NAME =''

# ---------------- SIGNIFICATIVITÀ / SECOND RUN ----------------
PVALUE_THRESHOLD = 0.05    # soglia per considerare significativo
N_PERM_HIGH = 10000        # numero permutazioni per il secondo run

# GENE REDUCTION
VFC_FOLDERS = []
NULL_PRECENTAGE = 0.30
OUTPUT_FOLDER = ''