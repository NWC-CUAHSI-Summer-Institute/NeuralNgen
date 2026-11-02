import re, sys
from pathlib import Path

MARKER = "# patched_by_patch_existing_configs"

def _sub_key(key, new_value, text):
    pat = re.compile(rf"(\b{re.escape(key)}\b\s*:\s*)(\"[^\"]*\"|\S+)")
    return pat.subn(lambda m: m.group(1) + new_value, text)

def _insert_after_key(key, block, text):
    pat = re.compile(rf"^([ \t]*\b{re.escape(key)}\b\s*:.*)$", re.MULTILINE)
    return pat.subn(lambda m: m.group(1) + "\n" + block, text, count=1)

def patch_one(p, scenario):
    text = p.read_text()
    if MARKER in text:
        return "already patched"
    text, n1 = _sub_key("binary_nexus_file_folder", "/ngen/qlat_cache/", text)
    if n1 == 0:
        return "SKIPPED - no binary_nexus_file_folder key"
    if "stream_output:" not in text:
        block = ("    stream_output:\n"
                 f"        stream_output_directory: ./outputs/troute_{scenario}_stream/\n"
                 "        stream_output_time: -1\n")
        text, n2 = _insert_after_key("lakeout_output", block, text)
        if n2 == 0:
            return "SKIPPED - no lakeout_output anchor"
    p.write_text(text + f"\n{MARKER}\n")
    return "patched"

root = Path(sys.argv[1])
counts = {}
for gd in sorted(root.glob("gage-*")):
    for p in gd.glob("config/*/troute_*.yaml"):
        sc = p.parent.name
        st = patch_one(p, sc)
        counts[st] = counts.get(st, 0) + 1
        if st.startswith("SKIPPED"):
            print(gd.name, sc, st)
        if st in ("patched", "already patched"):
            for d in (f"ngen_{sc}", f"troute_{sc}", f"troute_{sc}_stream"):
                (gd / "outputs" / d).mkdir(parents=True, exist_ok=True)
print(counts)
