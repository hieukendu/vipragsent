"""Export only manifest-listed release files, leaving the working data untouched."""
import hashlib
import json
import zipfile
from pathlib import Path

from verify_release import main as verify_release

ROOT = Path(__file__).resolve().parent


def main():
    verify_release()
    manifest = json.loads((ROOT/"FILE_MANIFEST.json").read_text())
    entries = {e["path"]:e["sha256"] for e in manifest["files"]}
    for name in ("FILE_MANIFEST.json","CHECKSUMS.sha256"):
        entries[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    target = ROOT/"submission/ARR_clean_anonymous.zip"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(entries):
                path = (ROOT/name).resolve()
                if not path.is_relative_to(ROOT) or path.is_symlink():
                    raise ValueError("Unsafe release member")
                # Fixed attributes avoid leaking host usernames or file timestamps.
                info = zipfile.ZipInfo("ARR_clean/"+name,date_time=(2026,10,9,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.create_system=0
                info.external_attr=0
                archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or set(archive.namelist())!={"ARR_clean/"+n for n in entries}:
            raise ValueError("ZIP content mismatch")
        if any(hashlib.sha256(archive.read("ARR_clean/"+n)).hexdigest()!=v for n,v in entries.items()):
            raise ValueError("Existing export differs; not overwritten")
    print(json.dumps({"status":"PASS","archive":target.relative_to(ROOT).as_posix(),"files":len(entries),"legacy_and_private_material_included":False}))


if __name__=="__main__":
    main()
