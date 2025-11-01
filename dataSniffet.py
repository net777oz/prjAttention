import csv

fname = "data/varA_.csv"
with open(fname, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    for li, row in enumerate(reader, start=1):
        for ci, val in enumerate(row, start=1):
            v = val.strip()
            try:
                float(v)
            except ValueError:
                print(f"비숫자 값 발견 → line {li}, col {ci}, value='{val}'")

