from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

RID = f"{{{NS['r']}}}id"
EMBED = f"{{{NS['r']}}}embed"
LINK = f"{{{NS['r']}}}link"


def xml_root(zf: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError:
        return None


def rels_path(part: str) -> str:
    folder, filename = posixpath.split(part)
    return posixpath.join(folder, "_rels", filename + ".rels")


def normalize_target(part: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(part), target))


def relationships(zf: zipfile.ZipFile, part: str) -> dict[str, dict[str, str]]:
    root = xml_root(zf, rels_path(part))
    if root is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for rel in root.findall("pr:Relationship", NS):
        rid = rel.get("Id", "")
        target = rel.get("Target", "")
        mode = rel.get("TargetMode", "Internal")
        out[rid] = {
            "type": rel.get("Type", ""),
            "target": target if mode == "External" else normalize_target(part, target),
            "mode": mode,
        }
    return out


def paragraph_texts(node: ET.Element) -> list[str]:
    paras: list[str] = []
    for p in node.findall(".//a:p", NS):
        pieces: list[str] = []
        for el in p.iter():
            if el.tag in (f"{{{NS['a']}}}t", f"{{{NS['m']}}}t") and el.text:
                pieces.append(el.text)
            elif el.tag == f"{{{NS['a']}}}tab":
                pieces.append("\t")
            elif el.tag == f"{{{NS['a']}}}br":
                pieces.append("\n")
        text = "".join(pieces).strip()
        if text:
            paras.append(text)
    return paras


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


URL_RE = re.compile(r"https?://[^\s<>\]\[)\}\"']+", re.I)
DOI_RE = re.compile(r"(?:doi\s*:\s*|https?://doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
SOURCE_WORD_RE = re.compile(
    r"\b(source|sources|reference|references|bibliography|adapted from|available at|retrieved from|figure from|image from|accessed)\b",
    re.I,
)


def reference_candidates(lines: list[str], external_links: list[str]) -> dict[str, list[str]]:
    urls: list[str] = list(external_links)
    dois: list[str] = []
    source_lines: list[str] = []
    for line in lines:
        urls.extend(URL_RE.findall(line))
        dois.extend(m.group(1).rstrip(".,;") for m in DOI_RE.finditer(line))
        if SOURCE_WORD_RE.search(line) or URL_RE.search(line) or DOI_RE.search(line):
            source_lines.append(line)
    return {
        "urls": ordered_unique(urls),
        "dois": ordered_unique(dois),
        "source_lines": ordered_unique(source_lines),
    }


def detect_title(root: ET.Element) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for sp in root.findall(".//p:sp", NS):
        paras = paragraph_texts(sp)
        if not paras:
            continue
        ph = sp.find("p:nvSpPr/p:nvPr/p:ph", NS)
        ph_type = ph.get("type", "") if ph is not None else ""
        text = " ".join(paras).strip()
        if ph_type in {"title", "ctrTitle"}:
            return text, "placeholder"
        nv = sp.find("p:nvSpPr/p:cNvPr", NS)
        name = (nv.get("name", "") if nv is not None else "").lower()
        if "title" in name:
            candidates.append((text, "shape_name"))
    if candidates:
        return candidates[0]
    # Fallback: first short text block, excluding slide numbers and isolated punctuation.
    for sp in root.findall(".//p:sp", NS):
        paras = paragraph_texts(sp)
        if not paras:
            continue
        text = " ".join(paras).strip()
        if 2 <= len(text) <= 180 and not re.fullmatch(r"\d+", text):
            return text, "first_text"
    return "", "none"


def shape_records(root: ET.Element, rels: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sp in root.findall(".//p:sp", NS):
        nv = sp.find("p:nvSpPr/p:cNvPr", NS)
        ph = sp.find("p:nvSpPr/p:nvPr/p:ph", NS)
        paras = paragraph_texts(sp)
        rec = {
            "kind": "text_shape",
            "id": nv.get("id", "") if nv is not None else "",
            "name": nv.get("name", "") if nv is not None else "",
            "title": nv.get("title", "") if nv is not None else "",
            "descr": nv.get("descr", "") if nv is not None else "",
            "placeholder": ph.get("type", "") if ph is not None else "",
            "text": paras,
        }
        if paras or rec["descr"] or rec["title"]:
            records.append(rec)

    for pic in root.findall(".//p:pic", NS):
        nv = pic.find("p:nvPicPr/p:cNvPr", NS)
        blip = pic.find("p:blipFill/a:blip", NS)
        rid = (blip.get(EMBED) or blip.get(LINK) or "") if blip is not None else ""
        rel = rels.get(rid, {})
        records.append(
            {
                "kind": "picture",
                "id": nv.get("id", "") if nv is not None else "",
                "name": nv.get("name", "") if nv is not None else "",
                "title": nv.get("title", "") if nv is not None else "",
                "descr": nv.get("descr", "") if nv is not None else "",
                "target": rel.get("target", ""),
            }
        )

    for gf in root.findall(".//p:graphicFrame", NS):
        nv = gf.find("p:nvGraphicFramePr/p:cNvPr", NS)
        gd = gf.find("a:graphic/a:graphicData", NS)
        uri = gd.get("uri", "") if gd is not None else ""
        kind = "graphic_frame"
        target = ""
        if gd is not None and gd.find("a:tbl", NS) is not None:
            kind = "table"
        elif gd is not None and gd.find("c:chart", NS) is not None:
            kind = "chart"
            chart_el = gd.find("c:chart", NS)
            rid = chart_el.get(RID, "") if chart_el is not None else ""
            target = rels.get(rid, {}).get("target", "")
        elif "diagram" in uri:
            kind = "smartart"
        records.append(
            {
                "kind": kind,
                "id": nv.get("id", "") if nv is not None else "",
                "name": nv.get("name", "") if nv is not None else "",
                "title": nv.get("title", "") if nv is not None else "",
                "descr": nv.get("descr", "") if nv is not None else "",
                "target": target,
                "text": paragraph_texts(gf),
            }
        )
    return records


def external_links(root: ET.Element, rels: dict[str, dict[str, str]]) -> list[str]:
    links: list[str] = []
    for el in root.iter():
        for attr in (RID, f"{{{NS['r']}}}embed", f"{{{NS['r']}}}link"):
            rid = el.get(attr)
            if rid and rid in rels and rels[rid]["mode"] == "External":
                links.append(rels[rid]["target"])
    return ordered_unique(links)


def chart_summary(zf: zipfile.ZipFile, chart_part: str) -> dict[str, Any]:
    root = xml_root(zf, chart_part)
    if root is None:
        return {"part": chart_part, "text": [], "series": [], "values": []}
    text = ordered_unique([t.text.strip() for t in root.findall(".//a:t", NS) if t.text and t.text.strip()])
    series: list[dict[str, Any]] = []
    for ser in root.findall(".//c:ser", NS):
        tx = ordered_unique([x.text.strip() for x in ser.findall(".//c:tx//c:v", NS) if x.text and x.text.strip()])
        cats = ordered_unique([x.text.strip() for x in ser.findall(".//c:cat//c:v", NS) if x.text and x.text.strip()])
        vals = [x.text.strip() for x in ser.findall(".//c:val//c:v", NS) if x.text and x.text.strip()]
        series.append({"name": tx, "categories": cats[:30], "values": vals[:30]})
    return {"part": chart_part, "text": text, "series": series}


def image_meta(zf: zipfile.ZipFile, target: str) -> dict[str, Any]:
    try:
        data = zf.read(target)
    except KeyError:
        return {"part": target, "missing": True}
    return {
        "part": target,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "extension": posixpath.splitext(target)[1].lower(),
    }


def slide_order(zf: zipfile.ZipFile) -> list[str]:
    pres_part = "ppt/presentation.xml"
    root = xml_root(zf, pres_part)
    if root is None:
        return []
    rels = relationships(zf, pres_part)
    out: list[str] = []
    for sldid in root.findall(".//p:sldId", NS):
        rid = sldid.get(RID, "")
        if rid in rels:
            out.append(rels[rid]["target"])
    return out


def extract_deck(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        slides: list[dict[str, Any]] = []
        media_index: dict[str, dict[str, Any]] = {}
        for number, slide_part in enumerate(slide_order(zf), start=1):
            root = xml_root(zf, slide_part)
            if root is None:
                continue
            rels = relationships(zf, slide_part)
            title, title_method = detect_title(root)
            all_text = paragraph_texts(root)
            shapes = shape_records(root, rels)
            links = external_links(root, rels)

            note_lines: list[str] = []
            note_links: list[str] = []
            notes_part = ""
            for rel in rels.values():
                if rel["type"].endswith("/notesSlide"):
                    notes_part = rel["target"]
                    notes_root = xml_root(zf, notes_part)
                    if notes_root is not None:
                        note_lines = paragraph_texts(notes_root)
                        note_links = external_links(notes_root, relationships(zf, notes_part))
                    break

            chart_parts = ordered_unique([s.get("target", "") for s in shapes if s["kind"] == "chart" and s.get("target")])
            charts = [chart_summary(zf, cp) for cp in chart_parts]

            picture_parts = ordered_unique([s.get("target", "") for s in shapes if s["kind"] == "picture" and s.get("target")])
            for pp in picture_parts:
                media_index.setdefault(pp, image_meta(zf, pp))

            lines_for_refs = all_text + note_lines + [s.get("descr", "") for s in shapes] + [s.get("title", "") for s in shapes]
            refs = reference_candidates(lines_for_refs, links + note_links)
            counts = Counter(s["kind"] for s in shapes)
            slides.append(
                {
                    "slide": number,
                    "part": slide_part,
                    "title": title,
                    "title_method": title_method,
                    "text": all_text,
                    "notes": note_lines,
                    "notes_part": notes_part,
                    "references": refs,
                    "object_counts": dict(counts),
                    "objects": shapes,
                    "charts": charts,
                }
            )

        return {
            "file": str(path),
            "filename": path.name,
            "bytes": path.stat().st_size,
            "slide_count": len(slides),
            "slides": slides,
            "media": list(media_index.values()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    decks = [extract_deck(p) for p in sorted(args.input_dir.rglob("*.pptx"), key=lambda p: p.name.lower())]
    payload = {
        "input_dir": str(args.input_dir),
        "deck_count": len(decks),
        "total_slides": sum(d["slide_count"] for d in decks),
        "decks": decks,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    slides_tsv = args.output_json.with_name("all_slide_index.tsv")
    links_tsv = args.output_json.with_name("external_links.tsv")
    with slides_tsv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["deck", "slide", "title", "title_method", "text", "speaker_notes", "pictures", "tables", "charts"])
        for deck in decks:
            for slide in deck["slides"]:
                writer.writerow(
                    [
                        deck["filename"],
                        slide["slide"],
                        slide["title"],
                        slide["title_method"],
                        " | ".join(slide["text"]),
                        " | ".join(slide["notes"]),
                        slide["object_counts"].get("picture", 0),
                        slide["object_counts"].get("table", 0),
                        slide["object_counts"].get("chart", 0),
                    ]
                )
    with links_tsv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["deck", "slide", "title", "url"])
        for deck in decks:
            for slide in deck["slides"]:
                for url in slide["references"]["urls"]:
                    if url and url != "NULL":
                        writer.writerow([deck["filename"], slide["slide"], slide["title"], url])
    print(
        json.dumps(
            {
                "deck_count": payload["deck_count"],
                "total_slides": payload["total_slides"],
                "output": str(args.output_json),
                "slide_index": str(slides_tsv),
                "links": str(links_tsv),
            }
        )
    )


if __name__ == "__main__":
    main()
