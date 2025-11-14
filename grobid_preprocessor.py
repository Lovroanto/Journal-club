import os
import requests
from pathlib import Path

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

def preprocess_pdf(pdf_path, output_dir):
    """
    Minimal GROBID wrapper:
    - Sends PDF to GROBID
    - Saves raw TEI XML (the true output!)
    - Saves raw text extracted by GROBID (optional)
    - No parsing, no splitting, no chunking
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📤 Sending PDF to GROBID:\n   {pdf_path}")

    with pdf_path.open("rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            data={"consolidateCitations": "1", "includeRawCitations": "1"},
            timeout=600
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"GROBID returned HTTP {response.status_code}\n"
            f"Message: {response.text[:500]}"
        )

    tei_xml = response.text

    # ---------------------------------------
    # Save raw TEI XML
    # ---------------------------------------
    raw_tei_path = output_dir / "raw_grobid_tei.xml"
    raw_tei_path.write_text(tei_xml, encoding="utf-8")

    # ---------------------------------------
    # Save raw text (optional)
    # ---------------------------------------
    try:
        # A crude extraction from TEI tags <p>, <figure>, etc.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(tei_xml, "lxml-xml")
        raw_text = "\n".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
    except Exception:
        raw_text = ""

    raw_text_path = output_dir / "raw_grobid_text.txt"
    raw_text_path.write_text(raw_text, encoding="utf-8")

    print("✅ Done! Raw GROBID output saved:")
    print(f"   TEI XML : {raw_tei_path}")
    print(f"   Text    : {raw_text_path}")

    return {
        "raw_tei": str(raw_tei_path),
        "raw_text": str(raw_text_path),
    }
