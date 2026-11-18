"""
convert_report_to_html.py
-------------------------
Converts docs/PathSnap_Consolidated_Technical_Report.md into a standalone,
professionally styled HTML file with embedded GitHub Markdown CSS, table styling,
and print-to-PDF optimizations.
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
        margin: 20mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        color: #24292e;
        background-color: #ffffff;
        max-width: 960px;
        margin: 0 auto;
        padding: 40px 20px;
    }}
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
        color: #1a1e22;
    }}
    h1 {{
        font-size: 2em;
        border-bottom: 2px solid #eaecef;
        padding-bottom: 0.3em;
        text-align: center;
        color: #0366d6;
    }}
    h2 {{
        font-size: 1.5em;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
        margin-top: 32px;
        color: #24292e;
    }}
    h3 {{
        font-size: 1.25em;
        margin-top: 24px;
    }}
    h4 {{
        font-size: 1.1em;
    }}
    p, ul, ol, table, pre, blockquote {{
        margin-top: 0;
        margin-bottom: 16px;
    }}
    ul, ol {{
        padding-left: 2em;
    }}
    li + li {{
        margin-top: 0.25em;
    }}
    code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 85%;
        background-color: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        color: #032f62;
    }}
    pre {{
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 16px;
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
        margin-bottom: 20px;
        display: block;
        overflow-x: auto;
    }}
    table th, table td {{
        padding: 8px 12px;
        border: 1px solid #dfe2e5;
        text-align: left;
    }}
    table th {{
        background-color: #f6f8fa;
        font-weight: 600;
    }}
    table tr:nth-child(2n) {{
        background-color: #fafbfc;
    }}
    blockquote {{
        padding: 0 1em;
        color: #6a737d;
        border-left: 0.25em solid #dfe2e5;
        background-color: #f9f9f9;
        padding: 8px 16px;
        border-radius: 0 4px 4px 0;
    }}
    hr {{
        height: 0.25em;
        padding: 0;
        margin: 24px 0;
        background-color: #e1e4e8;
        border: 0;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: bold;
        border-radius: 12px;
        background: #e1f5fe;
        color: #0277bd;
    }}
    @media print {{
        body {{
            max-width: 100%;
            padding: 0;
            font-size: 11pt;
        }}
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
        table, pre {{
            page-break-inside: avoid;
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
