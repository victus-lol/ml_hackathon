"""
convert_report_to_docx.py
-------------------------
Converts docs/PathSnap_Consolidated_Technical_Report.md into a beautifully styled,
professional Microsoft Word document (.docx) with embedded tables, figures,
captions, and executive formatting.
"""

import re
import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

REPO_ROOT = Path(r"c:\Users\adity\Downloads\ML HACKATHON\ml_hackathon")
MD_PATH = REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.md"
DOCX_PATH = REPO_ROOT / "docs" / "PathSnap_Consolidated_Technical_Report.docx"
DOWNLOADS_DOCX = Path(r"c:\Users\adity\Downloads\PathSnap_Consolidated_Technical_Report.docx")
IMG_DIR = REPO_ROOT / "docs"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:insideV w:val="none"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_styled_paragraph(doc, text, style='Normal', space_after=6, space_before=0, line_spacing=1.15):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = line_spacing
    
    # Simple markdown inline formatting: **bold**, *italic*, `code`
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = p.add_run(token[2:-2])
            run.bold = True
        elif token.startswith('*') and token.endswith('*') and not token.startswith('**'):
            run = p.add_run(token[1:-1])
            run.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run = p.add_run(token[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(11, 105, 163)
        else:
            # Clean math markers for clean DOCX display
            clean_token = token.replace('$', '').replace('\\mathbf{', '').replace('}', '').replace('\\text{', '').replace('\\perp', '⟂')
            p.add_run(clean_token)
    return p

def convert_md_to_docx():
    print(f"Reading markdown from: {MD_PATH}")
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    
    doc = Document()
    
    # Page setup - 0.75 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    # Styles setup
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Segoe UI'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(36, 41, 46)
    
    i = 0
    in_table = False
    table_rows = []
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for table
        if line.startswith('|') and line.endswith('|'):
            table_rows.append(line)
            i += 1
            continue
        elif table_rows:
            # Process accumulated table
            parse_and_add_table(doc, table_rows)
            table_rows = []
            
        if not line:
            i += 1
            continue
            
        # Title (H1)
        if line.startswith('# '):
            h1_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(h1_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(20)
            run.bold = True
            run.font.color.rgb = RGBColor(3, 102, 214)
            
        # H2
        elif line.startswith('## '):
            h2_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h2_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(14)
            run.bold = True
            run.font.color.rgb = RGBColor(26, 30, 34)
            
        # H3
        elif line.startswith('### '):
            h3_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h3_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(12)
            run.bold = True
            run.font.color.rgb = RGBColor(45, 55, 72)
            
        # H4
        elif line.startswith('#### '):
            h4_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h4_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(11)
            run.bold = True
            run.font.color.rgb = RGBColor(74, 85, 104)
            
        # Horizontal Rule
        elif line.startswith('---'):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run('—' * 55)
            run.font.color.rgb = RGBColor(220, 224, 230)
            
        # Images: ![alt](path)
        elif line.startswith('![') and '](' in line and line.endswith(')'):
            alt_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if alt_match:
                alt_text = alt_match.group(1)
                img_rel = alt_match.group(2)
                img_full = IMG_DIR / img_rel
                if img_full.exists():
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.paragraph_format.space_before = Pt(10)
                    p_img.paragraph_format.space_after = Pt(3)
                    p_img.paragraph_format.keep_with_next = True
                    try:
                        # Max 6.0 inches width
                        doc.add_picture(str(img_full), width=Inches(5.8))
                        # Center last picture
                        last_p = doc.paragraphs[-1]
                        last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        last_p.paragraph_format.space_after = Pt(2)
                    except Exception as e:
                        print(f"Error embedding picture {img_rel}: {e}")
                else:
                    print(f"Image not found: {img_full}")
                    
        # Figure Captions: *Figure X: ...*
        elif line.startswith('*Figure ') and line.endswith('*'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(line[1:-1])
            run.font.name = 'Segoe UI'
            run.font.size = Pt(9)
            run.italic = True
            run.font.color.rgb = RGBColor(100, 116, 139)
            
        # Blockquote / Alert callouts: > ...
        elif line.startswith('>'):
            callout_text = line.lstrip('> ').strip()
            if callout_text.startswith('[!NOTE]') or callout_text.startswith('[!IMPORTANT]') or callout_text.startswith('[!TIP]'):
                callout_title = callout_text.split(']')[0][2:]
                callout_body = callout_text.split(']')[-1].strip()
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                run_title = p.add_run(f"[{callout_title}] ")
                run_title.bold = True
                run_title.font.color.rgb = RGBColor(2, 119, 189)
                p.add_run(callout_body)
            else:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(4)
                run = p.add_run(callout_text)
                run.italic = True
                run.font.color.rgb = RGBColor(71, 85, 105)
                
        # Bullet list item
        elif line.startswith('* ') or line.startswith('- '):
            item_text = line[2:].strip()
            p = add_styled_paragraph(doc, item_text, space_after=3)
            p.paragraph_format.left_indent = Inches(0.25)
            # Add bullet symbol
            p.runs[0].text = "• " + p.runs[0].text if p.runs else "• "
            
        # Numbered list item
        elif re.match(r'^\d+\.\s', line):
            num_match = re.match(r'^(\d+\.\s)(.*)', line)
            num_prefix = num_match.group(1)
            item_text = num_match.group(2)
            p = add_styled_paragraph(doc, item_text, space_after=3)
            p.paragraph_format.left_indent = Inches(0.25)
            p.runs[0].text = num_prefix + p.runs[0].text if p.runs else num_prefix
            
        # Normal Paragraph
        else:
            add_styled_paragraph(doc, line, space_after=6)
            
        i += 1
        
    if table_rows:
        parse_and_add_table(doc, table_rows)
        
    doc.save(str(DOCX_PATH))
    print(f"Successfully saved Word Document at: {DOCX_PATH}")
    
    # Copy to Downloads
    doc.save(str(DOWNLOADS_DOCX))
    print(f"Copied Word Document to: {DOWNLOADS_DOCX}")

def parse_and_add_table(doc, raw_rows):
    if len(raw_rows) < 2:
        return
    parsed = []
    for r in raw_rows:
        parts = [c.strip() for c in r.split('|')[1:-1]]
        # Filter separator row |--|--|
        if all(re.match(r'^:?-+:?$', c) for c in parts):
            continue
        parsed.append(parts)
        
    if not parsed:
        return
        
    num_cols = len(parsed[0])
    num_rows = len(parsed)
    
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    
    for r_idx, row_data in enumerate(parsed):
        row = table.rows[r_idx]
        is_header = (r_idx == 0)
        
        for c_idx, cell_value in enumerate(row_data):
            if c_idx >= num_cols:
                break
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            
            if is_header:
                set_cell_background(cell, "0366D6")
            elif r_idx % 2 == 1:
                set_cell_background(cell, "F8FAFC")
            else:
                set_cell_background(cell, "FFFFFF")
                
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            
            # Clean markdown inline formatting
            clean_text = cell_value.replace('**', '').replace('`', '').replace('$', '').replace('\\mathbf{', '').replace('}', '')
            run = p.add_run(clean_text)
            run.font.name = 'Segoe UI'
            run.font.size = Pt(8.5 if num_cols > 6 else 9.5)
            
            if is_header:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if '**' in cell_value:
                    run.bold = True
                run.font.color.rgb = RGBColor(30, 41, 59)
                
    # Add small spacing after table
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(2)
    p_spacer.paragraph_format.space_after = Pt(6)

if __name__ == "__main__":
    convert_md_to_docx()
