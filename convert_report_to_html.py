"""
convert_report_to_html.py
-------------------------
Converts docs/PathSnap_Consolidated_Technical_Report.md into a standalone,
professionally styled HTML file with embedded GitHub Markdown CSS, table styling,
image layout styling, and print-to-PDF optimizations.
"""

from pathlib import Path
import markdown

REPO_ROOT = Path(__file__).parent.resolve()
MD_PATH = REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.md"
HTML_PATH = REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PathSnap - Consolidated Technical Report</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 15mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 13.5px;
        line-height: 1.55;
        color: #24292e;
        background-color: #ffffff;
        max-width: 980px;
        margin: 0 auto;
        padding: 30px 25px;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 24px;
        margin-bottom: 14px;
        font-weight: 600;
        line-height: 1.25;
        color: #1a1e22;
    }}
    h1 {{
        font-size: 2.1em;
        border-bottom: 2px solid #0366d6;
        padding-bottom: 0.3em;
        text-align: center;
        color: #0366d6;
        margin-top: 10px;
    }}
    h2 {{
        font-size: 1.45em;
        border-bottom: 1.5px solid #eaecef;
        padding-bottom: 0.25em;
        margin-top: 28px;
        color: #1b2733;
    }}
    h3 {{
        font-size: 1.2em;
        margin-top: 20px;
        color: #2d3748;
    }}
    h4 {{
        font-size: 1.05em;
        color: #4a5568;
    }}
    p, ul, ol, table, pre, blockquote {{
        margin-top: 0;
        margin-bottom: 14px;
    }}
    ul, ol {{
        padding-left: 2em;
    }}
    li + li {{
        margin-top: 0.25em;
    }}
    code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 88%;
        background-color: #f3f4f6;
        padding: 0.18em 0.4em;
        border-radius: 4px;
        color: #0b69a3;
    }}
    pre {{
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 14px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
        border: 1px solid #e1e4e8;
    }}
    pre code {{
        background-color: transparent;
        padding: 0;
        color: inherit;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0 22px 0;
        display: table;
        font-size: 12.5px;
    }}
    table th, table td {{
        padding: 7px 10px;
        border: 1px solid #dfe2e5;
        text-align: left;
    }}
    table th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #1e293b;
    }}
    table tr:nth-child(2n) {{
        background-color: #f8fafc;
    }}
    blockquote {{
        padding: 10px 16px;
        color: #2d3748;
        border-left: 4px solid #3b82f6;
        background-color: #f0f7ff;
        border-radius: 0 6px 6px 0;
        margin: 16px 0;
    }}
    img {{
        max-width: 95%;
        height: auto;
        display: block;
        margin: 20px auto;
        border-radius: 8px;
        border: 1px solid #dfe2e5;
        box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    }}
    hr {{
        height: 2px;
        padding: 0;
        margin: 26px 0;
        background-color: #e2e8f0;
        border: 0;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: bold;
        border-radius: 12px;
        background: #e1f5fe;
        color: #0277bd;
    }}
    @media print {{
        body {{
            max-width: 100%;
            padding: 0;
            font-size: 10.5pt;
            line-height: 1.45;
        }}
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
        table, pre, img, blockquote {{
            page-break-inside: avoid;
        }}
        img {{
            max-width: 90% !important;
            margin: 14px auto;
            box-shadow: none;
        }}
        table {{
            font-size: 9.5pt;
        }}
    }}
</style>
</head>
<body>
{content}
</body>
</html>
"""

def convert():
    if not MD_PATH.exists():
        print(f"Error: {MD_PATH} not found.")
        return
    text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'codehilite'])
    full_html = HTML_TEMPLATE.format(content=html_body)
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Successfully generated HTML report at: {HTML_PATH.resolve()}")

if __name__ == "__main__":
    convert()
