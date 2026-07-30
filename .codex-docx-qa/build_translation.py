from __future__ import annotations

import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree


SOURCE = Path(r"C:\Users\sergb\Desktop\PICL2\PICL2 LETTER.docx")
OUTPUT = Path(r"C:\Users\sergb\Desktop\PICL2\PICL2 LETTER - Latvian.docx")


def set_run_texts(paragraph, values: list[str]) -> None:
    if len(values) != len(paragraph.runs):
        raise ValueError(
            f"Unexpected run count in {paragraph.text!r}: "
            f"expected {len(values)}, got {len(paragraph.runs)}"
        )
    for run, value in zip(paragraph.runs, values):
        run.text = value
        if any(ord(char) > 127 for char in value):
            run.font.name = "Arial"
            rpr = run._element.get_or_add_rPr()
            rfonts = rpr.rFonts
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.insert(0, rfonts)
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rfonts.set(qn(attr), "Arial")
            lang = rpr.find(qn("w:lang"))
            if lang is None:
                lang = OxmlElement("w:lang")
                rpr.append(lang)
            lang.set(qn("w:val"), "lv-LV")


def patch_xml_text(docx_path: Path) -> None:
    replacements = {
        "Regenerative Medicine and Interventional Orthopedics":
            "Reģeneratīvā medicīna un intervencionālā ortopēdija",
        "Centeno-Schultz Clinic ": "Centeno-Schultz klīnika ",
        "phone  ": "tālr.  ",
        "fax": "fakss",
    }
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    with ZipFile(docx_path, "r") as src, tempfile.NamedTemporaryFile(
        suffix=".docx", delete=False, dir=docx_path.parent
    ) as tmp_handle:
        tmp_path = Path(tmp_handle.name)

    try:
        with ZipFile(docx_path, "r") as src, ZipFile(
            tmp_path, "w", compression=ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename in {"word/header2.xml", "word/footer2.xml"}:
                    root = etree.fromstring(data)
                    for text_node in root.xpath(".//w:t", namespaces=ns):
                        if text_node.text in replacements:
                            text_node.text = replacements[text_node.text]
                        if text_node.text and any(
                            ord(char) > 127 for char in text_node.text
                        ):
                            run = text_node.getparent().getparent()
                            rpr = run.find(qn("w:rPr"))
                            if rpr is None:
                                rpr = OxmlElement("w:rPr")
                                run.insert(0, rpr)
                            rfonts = rpr.find(qn("w:rFonts"))
                            if rfonts is None:
                                rfonts = OxmlElement("w:rFonts")
                                rpr.insert(0, rfonts)
                            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                                rfonts.set(qn(attr), "Arial")
                            lang = rpr.find(qn("w:lang"))
                            if lang is None:
                                lang = OxmlElement("w:lang")
                                rpr.append(lang)
                            lang.set(qn("w:val"), "lv-LV")
                    data = etree.tostring(
                        root, xml_declaration=True, encoding="UTF-8", standalone="yes"
                    )
                dst.writestr(item, data)
        os.replace(tmp_path, docx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build() -> None:
    document = Document(SOURCE)
    paragraphs = document.paragraphs

    set_run_texts(paragraphs[0], ["Vārds, uzvārds:  ", "", "", "Sergejs Burtovojs "])
    set_run_texts(paragraphs[1], ["Dzimšanas datums:  ", "", "", "11/01/1988"])
    set_run_texts(paragraphs[2], ["Datums:", "\t", "07", "/", "15", "/2025"])
    set_run_texts(paragraphs[4], ["Visām ieinteresētajām personām, "])
    set_run_texts(
        paragraphs[6],
        [
            "Sergejam ",
            "06/29/2026",
            "",
            "",
            "",
            "",
            " mūsu klīnikā tika veikta procedūra. Procedūra noritēja ļoti labi, "
            "un nopietnu bažu nebija. "
            "Šīs procedūras ierastais atveseļošanās periods ir ",
            "aptuveni viens mēnesis",
            "; šajā laikā iesakām ierobežot smagumu celšanu, pārmaiņus sēdēt un "
            "stāvēt, kā arī līdz minimumam samazināt datora lietošanu.",
        ],
    )
    set_run_texts(
        paragraphs[7],
        [
            "Fizioterapija ieteicama ",
            "1–2 reizes nedēļā 12 nedēļu garumā",
            ", galveno uzmanību pievēršot speciālista vadītai rehabilitācijai un "
            "pakāpeniskai spēka atjaunošanai.",
        ],
    )
    set_run_texts(
        paragraphs[8],
        [
            "Ja jums rodas jautājumi, lūdzu, sazinieties ar mūsu klīniku pa tālruni ",
            "303-429-6448",
            ".",
        ],
    )
    set_run_texts(paragraphs[16], ["Centeno-Schultz klīnika", "\t"])
    set_run_texts(paragraphs[19], ["Tālr.: 303-429-6448"])
    set_run_texts(paragraphs[20], ["Fakss: 303-429-6373"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    patch_xml_text(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
