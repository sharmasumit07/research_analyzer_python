from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import fitz  # PyMuPDF
import requests
import os
from PIL import Image
import io
import base64
import json
import tempfile
import shutil
from pathlib import Path
import time
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from flask import Flask, request, jsonify, send_file
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# OpenCV / numpy for image analysis
import cv2
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEMP_DIR = Path("temp_images")
REPORTS_DIR = Path("generated_reports")
TEMP_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Helpers: OpenRouter call with better error handling
# ---------------------------
def call_openrouter(prompt, model="openai/gpt-3.5-turbo"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert AI research paper reviewer and figure interpreter."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, json=data, timeout=30)
        if response.status_code != 200:
            return f"[API Error {response.status_code}] {response.text}"
        
        result = response.json()
        if 'choices' not in result:
            return f"[API Response Error] Unexpected format: {result}"
        
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"[Request Error] {str(e)}"

# ---------------------------
# Enhanced PDF text extraction with metadata
# ---------------------------
def extract_text_and_metadata_from_pdf(pdf_file):
    if isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)
    else:
        doc = fitz.open(pdf_file.name)
        
    text = ""
    metadata = {
        "total_pages": len(doc),
        "title": doc.metadata.get("title", "Unknown"),
        "author": doc.metadata.get("author", "Unknown"),
        "subject": doc.metadata.get("subject", "Unknown"),
        "creator": doc.metadata.get("creator", "Unknown"),
        "page_texts": []
    }
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        metadata["page_texts"].append({
            "page": page_num + 1,
            "text_length": len(page_text),
            "has_images": len(page.get_images()) > 0
        })
    
    return text, metadata

# ---------------------------
# Enhanced image extraction with better file management
# ---------------------------
def extract_images_from_pdf(pdf_file, min_width=80, min_height=80):
    if isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)
    else:
        doc = fitz.open(pdf_file.name)
        
    image_list = []
    
    # Create session-specific temp directory
    session_dir = TEMP_DIR / f"session_{int(time.time())}"
    session_dir.mkdir(exist_ok=True)
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                if img_pil.width >= min_width and img_pil.height >= min_height:
                    temp_path = session_dir / f"img_p{page_index+1}_i{img_index+1}.png"
                    img_pil.save(temp_path, format="PNG")
                    
                    image_info = {
                        "path": str(temp_path),
                        "page": page_index + 1,
                        "index": img_index + 1,
                        "width": img_pil.width,
                        "height": img_pil.height,
                        "format": base_image.get("ext", "unknown")
                    }
                    image_list.append(image_info)
                    
            except Exception as e:
                print(f"Error extracting image on page {page_index+1}, index {img_index+1}: {e}")
                continue
    
    return image_list, str(session_dir)

# ---------------------------
# Enhanced image analysis with more features
# ---------------------------
def get_dominant_colors_cv(img_bgr, k=3):
    """Return list of k dominant colors in hex using kmeans."""
    img_small = cv2.resize(img_bgr, (max(32, img_bgr.shape[1]//4), max(32, img_bgr.shape[0]//4)))
    data = img_small.reshape((-1,3)).astype(np.float32)
    
    if data.shape[0] < k:
        colors = [tuple(int(c) for c in img_bgr[0,0][::-1])]
        return [f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in colors]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 8
    
    try:
        _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(int)
        hex_colors = []
        for c in centers:
            b, g, r = int(c[0]), int(c[1]), int(c[2])
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return hex_colors
    except Exception:
        b, g, r = int(img_bgr[0,0,0]), int(img_bgr[0,0,1]), int(img_bgr[0,0,2])
        return [f"#{r:02x}{g:02x}{b:02x}"]

def analyze_image_advanced(image_path):
    """Enhanced image analysis with more chart type detection."""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "cannot_read_image"}
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    
    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(50, min(w,h)//20), 
                           minLineLength=min(w,h)//15, maxLineGap=10)
    line_count = 0 if lines is None else len(lines)
    
    # Analyze line orientations
    horiz_count = vert_count = diag_count = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            
            if abs(angle) <= 15 or abs(angle) >= 165:  # Horizontal
                horiz_count += 1
            elif 75 <= abs(angle) <= 105:  # Vertical
                vert_count += 1
            else:  # Diagonal
                diag_count += 1
    
    # Contour analysis
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rect_like = tall_bars = wide_bars = 0
    areas = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(100, (w*h)*0.0005):
            continue
        
        areas.append(area)
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = (hh+1) / (ww+1)
        
        if aspect > 2 and hh > h*0.1:  # Tall bars
            tall_bars += 1
        elif aspect < 0.5 and ww > w*0.1:  # Wide bars
            wide_bars += 1
        elif 0.7 <= aspect <= 1.4:  # Square-ish
            rect_like += 1
    
    # Circle detection for pie charts
    try:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, 
                                 minDist=min(w,h)//8, param1=50, param2=30,
                                 minRadius=min(w,h)//20, maxRadius=min(w,h)//3)
        circle_count = 0 if circles is None else circles.shape[1]
    except:
        circle_count = 0
    
    # Text detection (for tables/labels)
    text_regions = 0
    try:
        # Simple text detection using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        text_contours, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in text_contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if 5 <= hh <= 30 and ww > hh:  # Text-like dimensions
                text_regions += 1
    except:
        pass
    
    # Color analysis
    colors = get_dominant_colors_cv(img, k=4)
    
    # Enhanced chart type classification
    chart_type = classify_chart_type(line_count, horiz_count, vert_count, tall_bars, 
                                   wide_bars, circle_count, text_regions, edge_density)
    
    return {
        "dimensions": {"width": w, "height": h},
        "edge_density": round(edge_density, 4),
        "lines": {
            "total": line_count,
            "horizontal": horiz_count,
            "vertical": vert_count,
            "diagonal": diag_count
        },
        "shapes": {
            "rectangular_regions": rect_like,
            "tall_bars": tall_bars,
            "wide_bars": wide_bars,
            "circles": circle_count,
            "text_regions": text_regions
        },
        "colors": {
            "dominant_colors": colors,
            "color_count": len(set(colors))
        },
        "classification": {
            "likely_type": chart_type,
            "confidence_indicators": get_confidence_indicators(chart_type, tall_bars, wide_bars, 
                                                             circle_count, line_count, text_regions)
        }
    }

def classify_chart_type(line_count, horiz_count, vert_count, tall_bars, wide_bars, 
                       circle_count, text_regions, edge_density):
    """Enhanced chart type classification."""
    if circle_count >= 1 and tall_bars <= 2:
        return "pie chart or circular diagram"
    elif tall_bars >= 3:
        return "vertical bar chart"
    elif wide_bars >= 3:
        return "horizontal bar chart"
    elif line_count > 15 and edge_density > 0.1 and tall_bars <= 1:
        return "line chart or plot"
    elif horiz_count >= 5 and vert_count >= 5 and text_regions > 10:
        return "table or grid"
    elif edge_density < 0.05:
        return "photograph or continuous image"
    elif text_regions > 20:
        return "text-heavy diagram or flowchart"
    else:
        return "complex diagram or mixed visualization"

def get_confidence_indicators(chart_type, tall_bars, wide_bars, circles, lines, text_regions):
    """Provide confidence indicators for chart type classification."""
    indicators = []
    
    if "bar chart" in chart_type:
        indicators.append(f"{max(tall_bars, wide_bars)} bar-like shapes detected")
    if "pie chart" in chart_type:
        indicators.append(f"{circles} circular regions found")
    if "line chart" in chart_type:
        indicators.append(f"{lines} line segments detected")
    if "table" in chart_type:
        indicators.append(f"{text_regions} text regions found")
    
    return indicators

# ---------------------------
# Enhanced caption generation focused on meaning and insights
# ---------------------------
def generate_meaningful_caption(image_info, analysis_obs, context=""):
    """Generate meaningful caption focused on what the figure represents and its insights."""
    chart_type = analysis_obs['classification']['likely_type']
    colors = analysis_obs['colors']['dominant_colors'][:3]
    
    # Create context-aware prompt focused on meaning
    prompt = f"""
You are analyzing Figure {image_info['index']} from page {image_info['page']} of a research paper.

Chart Type Detected: {chart_type}
Surrounding Text Context: {context[:800]}

Your task is to interpret what this figure MEANS and REPRESENTS in the context of the research, not describe its visual properties.

Please provide a meaningful interpretation focusing on:

1. **What does this figure demonstrate or prove?** (What is the main finding or result shown?)
2. **What trends, patterns, or relationships does it reveal?** (What insights can be drawn?)
3. **How does this support or relate to the research hypothesis/question?** (What is its purpose in the paper?)

Guidelines:
- Focus on INSIGHTS, not visual descriptions
- Interpret the DATA and TRENDS shown
- Explain the SIGNIFICANCE to the research
- Avoid mentioning colors, shapes, sizes, or technical visual details
- Think like a domain expert interpreting results
- Be specific about what the data reveals

Provide 2-4 sentences of meaningful interpretation.
"""
    
    caption = call_openrouter(prompt)
    if caption.startswith("["):
        # Fallback based on chart type and context
        return generate_fallback_meaningful_caption(chart_type, context, image_info)
    
    return caption.strip()

def generate_fallback_meaningful_caption(chart_type, context, image_info):
    """Generate a meaningful fallback caption when API fails."""
    context_words = context.lower().split()
    
    # Common research terms to look for
    performance_terms = ['accuracy', 'performance', 'results', 'improvement', 'comparison', 'evaluation']
    trend_terms = ['trend', 'increase', 'decrease', 'correlation', 'relationship', 'over time']
    distribution_terms = ['distribution', 'frequency', 'proportion', 'percentage', 'ratio']
    
    if any(term in context_words for term in performance_terms):
        if 'bar chart' in chart_type:
            return f"Figure {image_info['index']} compares performance metrics across different methods or conditions, showing quantitative differences in the experimental results."
        elif 'line chart' in chart_type:
            return f"Figure {image_info['index']} demonstrates the progression of performance measures, revealing trends and improvements over the course of the study."
        elif 'pie chart' in chart_type:
            return f"Figure {image_info['index']} illustrates the relative contribution of different components to overall performance outcomes."
    
    elif any(term in context_words for term in trend_terms):
        return f"Figure {image_info['index']} reveals temporal patterns and relationships in the data, showing how key variables change and interact over the study period."
    
    elif any(term in context_words for term in distribution_terms):
        return f"Figure {image_info['index']} presents the distribution of observations across categories, highlighting the relative frequency and patterns in the dataset."
    
    # Generic meaningful interpretation based on chart type
    if 'bar chart' in chart_type:
        return f"Figure {image_info['index']} presents comparative analysis results, demonstrating quantitative differences between experimental conditions or methods."
    elif 'line chart' in chart_type:
        return f"Figure {image_info['index']} illustrates temporal trends or progressive relationships, showing how variables evolve throughout the study."
    elif 'pie chart' in chart_type:
        return f"Figure {image_info['index']} shows the relative proportions and composition of different elements within the research framework."
    elif 'table' in chart_type:
        return f"Figure {image_info['index']} presents structured data and statistical summaries that support the research findings and methodology."
    else:
        return f"Figure {image_info['index']} provides visual evidence supporting key research conclusions and demonstrates relationships between study variables."

# ---------------------------
# PDF Report Generation
# ---------------------------
def generate_pdf_report(text_analysis, image_infos, captions, analyses, metadata, output_path):
    """Generate a comprehensive PDF report of the analysis."""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkgreen,
        spaceBefore=20,
        spaceAfter=10
    )
    
    # Title
    story.append(Paragraph("AI Research Paper Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Document metadata
    story.append(Paragraph("Document Information", heading_style))
    metadata_table_data = [
        ["Property", "Value"],
        ["Title", metadata.get('title', 'Unknown')],
        ["Author", metadata.get('author', 'Unknown')],
        ["Total Pages", str(metadata['total_pages'])],
        ["Figures Extracted", str(len(image_infos))],
        ["Creator", metadata.get('creator', 'Unknown')]
    ]
    
    metadata_table = Table(metadata_table_data, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # Text analysis
    story.append(Paragraph("Research Analysis & Insights", heading_style))
    # Split text analysis into paragraphs
    analysis_paragraphs = text_analysis.split('\n\n')
    for para in analysis_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 10))
    
    story.append(Spacer(1, 30))
    
    # Figure analysis
    if image_infos:
        story.append(Paragraph("Figure Interpretations & Insights", heading_style))
        
        for i, (img_info, caption, analysis) in enumerate(zip(image_infos, captions, analyses)):
            # Figure heading
            fig_title = f"Figure {img_info['index']} (Page {img_info['page']})"
            story.append(Paragraph(fig_title, styles['Heading3']))
            
            # Add image if it exists
            try:
                if os.path.exists(img_info['path']):
                    # Resize image to fit page
                    img_width = min(4*inch, img_info['width'] * 0.5)
                    img_height = img_info['height'] * (img_width / img_info['width'])
                    
                    if img_height > 3*inch:
                        img_height = 3*inch
                        img_width = img_info['width'] * (img_height / img_info['height'])
                    
                    story.append(RLImage(img_info['path'], width=img_width, height=img_height))
                    story.append(Spacer(1, 10))
            except Exception as e:
                story.append(Paragraph(f"[Image could not be displayed]", styles['Italic']))
            
            # Figure interpretation (focus on meaning, not visual details)
            story.append(Paragraph(f"<b>Research Insight:</b> {caption}", styles['Normal']))
            
            # Chart type only (remove visual details)
            story.append(Paragraph(f"<b>Visualization Type:</b> {analysis['classification']['likely_type'].title()}", styles['Normal']))
            story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    return output_path

# ---------------------------
# Enhanced HTML generation with better styling
# ---------------------------
def create_enhanced_figure_display(image_infos, captions, analyses):
    """Create a rich HTML display for figures with detailed information."""
    if not image_infos:
        return "<div class='no-figures'><p>No figures detected in this PDF.</p></div>"
    
    html_parts = ["""
    <style>
    .figures-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        padding: 10px;
    }
    .figure-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        max-width: 400px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .figure-image {
        max-width: 100%;
        height: auto;
        border: 1px solid #eee;
        border-radius: 4px;
    }
    .figure-info {
        margin-top: 10px;
        font-size: 12px;
        color: #666;
    }
    .figure-caption {
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.4;
        color: #333;
    }
    .chart-type {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin: 5px 0;
    }
    .colors {
        margin: 5px 0;
    }
    .color-swatch {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 3px;
        margin-right: 4px;
        border: 1px solid #ccc;
    }
    </style>
    <div class="figures-container">"""]
    
    for i, img_info in enumerate(image_infos):
        try:
            with open(img_info["path"], "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            data_uri = f"data:image/png;base64,{img_data}"
            
            analysis = analyses[i]
            caption = captions[i]
            
            # Create color swatches
            color_swatches = ""
            if "colors" in analysis and "dominant_colors" in analysis["colors"]:
                for color in analysis["colors"]["dominant_colors"][:4]:
                    color_swatches += f'<span class="color-swatch" style="background-color: {color}"></span>'
            
            html_parts.append(f"""
            <div class="figure-card">
                <img src="{data_uri}" alt="Figure {img_info['index']}" class="figure-image"/>
                <div class="figure-info">
                    <strong>Figure {img_info['index']}</strong> (Page {img_info['page']})
                    <div class="chart-type">{analysis['classification']['likely_type']}</div>
                </div>
                <div class="figure-caption"><strong>Research Insight:</strong> {caption}</div>
            </div>""")
        except Exception as e:
            html_parts.append(f"""
            <div class="figure-card">
                <p>Error displaying Figure {img_info['index']} from page {img_info['page']}: {e}</p>
            </div>""")
    
    html_parts.append("</div>")
    return "".join(html_parts)

# ---------------------------
# Text analysis enhancement
# ---------------------------
def analyze_paper_text(text, metadata):
    """Enhanced text analysis with better prompting."""
    prompt = f"""
Analyze this research paper with {metadata['total_pages']} pages:

Title: {metadata.get('title', 'Unknown')}
Author: {metadata.get('author', 'Unknown')}

Provide a structured analysis:

1. **Abstract Summary** (2-3 sentences)
2. **Main Research Question/Hypothesis**
3. **Methodology Overview** 
4. **Key Findings/Results**
5. **Strengths** (2-3 bullet points)
6. **Weaknesses/Limitations** (2-3 bullet points)
7. **Significance & Impact**
8. **Suggestions for Future Work**

TEXT CONTENT:
{text[:10000]}
"""
    
    analysis = call_openrouter(prompt, model="openai/gpt-3.5-turbo")
    return analysis

# ---------------------------
# Main analysis function
# ---------------------------
def analyze_pdf_comprehensive(pdf_file, progress=None):
    """Comprehensive PDF analysis with progress tracking."""
    try:
        if progress:
            progress(0.1, desc="Extracting text and metadata...")
        text, metadata = extract_text_and_metadata_from_pdf(pdf_file)
        
        if progress:
            progress(0.3, desc="Extracting images...")
        image_infos, session_dir = extract_images_from_pdf(pdf_file)
        
        if not image_infos:
            if progress:
                progress(0.8, desc="Analyzing text...")
            text_analysis = analyze_paper_text(text, metadata)
            return text_analysis, "<p>No figures found in this PDF.</p>", create_metadata_summary(metadata), []
        
        if progress:
            progress(0.5, desc="Analyzing figures...")
        analyses = []
        captions = []
        
        for i, img_info in enumerate(image_infos):
            if progress:
                progress(0.5 + (0.3 * i / len(image_infos)), desc=f"Analyzing figure {i+1}/{len(image_infos)}...")
            
            # Analyze image
            analysis = analyze_image_advanced(img_info["path"])
            analyses.append(analysis)
            
            # Generate meaningful caption with context
            relevant_text = get_relevant_text_context(text, img_info["page"])
            caption = generate_meaningful_caption(img_info, analysis, relevant_text)
            captions.append(caption)
        
        if progress:
            progress(0.9, desc="Creating final analysis...")
        
        # Generate comprehensive text analysis
        text_analysis = analyze_paper_text(text, metadata)
        
        # Create enhanced figure display
        figures_html = create_enhanced_figure_display(image_infos, captions, analyses)
        
        # Create metadata summary
        metadata_summary = create_metadata_summary(metadata, len(image_infos))
        
        if progress:
            progress(1.0, desc="Complete!")
        
        return text_analysis, figures_html, metadata_summary, {
            'image_infos': image_infos,
            'captions': captions,
            'analyses': analyses,
            'metadata': metadata
        }
        
    except Exception as e:
        return f"Error during analysis: {str(e)}", "<p>Error analyzing figures.</p>", f"Error: {str(e)}", {}

def get_relevant_text_context(full_text, page_num, context_length=800):
    """Extract relevant text context around a specific page with more content for better analysis."""
    lines = full_text.split('\n')
    page_marker = f"--- Page {page_num} ---"
    
    try:
        page_start = next(i for i, line in enumerate(lines) if page_marker in line)
        # Get more context - current page and surrounding pages
        context_start = max(0, page_start - 10)  # 10 lines before page
        context_end = min(len(lines), page_start + 50)  # 50 lines after page start
        page_text = '\n'.join(lines[context_start:context_end])
        return page_text[:context_length]
    except:
        # If page marker not found, search around the page number
        try:
            lines_per_page = len(lines) // max(1, int(page_num))
            estimated_start = max(0, (page_num - 1) * lines_per_page - 10)
            estimated_end = min(len(lines), page_num * lines_per_page + 30)
            return '\n'.join(lines[estimated_start:estimated_end])[:context_length]
        except:
            return ""

def create_metadata_summary(metadata, num_figures=0):
    """Create a summary of PDF metadata."""
    pages_with_images = sum(1 for p in metadata["page_texts"] if p["has_images"])
    
    return f"""
    **Document Information:**
    - **Title:** {metadata.get('title', 'Unknown')}
    - **Author:** {metadata.get('author', 'Unknown')}  
    - **Total Pages:** {metadata['total_pages']}
    - **Pages with Images:** {pages_with_images}
    - **Total Figures Extracted:** {num_figures}
    - **Creator:** {metadata.get('creator', 'Unknown')}
    """

# ---------------------------
# FastAPI Application for API
# ---------------------------
app = FastAPI(title="PDF Analyzer API", version="1.0.0")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],  # Add your Next.js URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_pdf_api(file: UploadFile = File(...)):
    """API endpoint to analyze PDF and return structured data."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        temp_file_path = TEMP_DIR / f"upload_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Analyze PDF
        text_analysis, figures_html, metadata_summary, analysis_data = analyze_pdf_comprehensive(str(temp_file_path))
        
        # Convert image paths to base64 for API response
        figures_data = []
        if analysis_data and 'image_infos' in analysis_data:
            for i, img_info in enumerate(analysis_data['image_infos']):
                try:
                    with open(img_info['path'], 'rb') as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode()
                    
                    figures_data.append({
                        'id': f"fig_{img_info['page']}_{img_info['index']}",
                        'page': img_info['page'],
                        'index': img_info['index'],
                        'image_data': f"data:image/png;base64,{img_base64}",
                        'caption': analysis_data['captions'][i],
                        'analysis': analysis_data['analyses'][i],
                        'dimensions': {
                            'width': img_info['width'],
                            'height': img_info['height']
                        }
                    })
                except Exception as e:
                    print(f"Error processing image {img_info['path']}: {e}")
        
        # Clean up temp file
        if temp_file_path.exists():
            os.remove(temp_file_path)
        
        return {
            "success": True,
            "data": {
                "text_analysis": text_analysis,
                "metadata": analysis_data.get('metadata', {}),
                "figures": figures_data,
                "summary": {
                    "total_pages": analysis_data.get('metadata', {}).get('total_pages', 0),
                    "total_figures": len(figures_data),
                    "document_title": analysis_data.get('metadata', {}).get('title', 'Unknown'),
                    "document_author": analysis_data.get('metadata', {}).get('author', 'Unknown')
                }
            }
        }
    
    except Exception as e:
        # Clean up temp file on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")

@app.post("/api/generate-report")
async def generate_report_api(file: UploadFile = File(...)):
    """API endpoint to analyze PDF and generate a downloadable report."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        temp_file_path = TEMP_DIR / f"upload_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Analyze PDF
        text_analysis, figures_html, metadata_summary, analysis_data = analyze_pdf_comprehensive(str(temp_file_path))
        
        # Generate PDF report
        report_filename = f"analysis_report_{int(time.time())}.pdf"
        report_path = REPORTS_DIR / report_filename
        
        if analysis_data:
            generate_pdf_report(
                text_analysis=text_analysis,
                image_infos=analysis_data.get('image_infos', []),
                captions=analysis_data.get('captions', []),
                analyses=analysis_data.get('analyses', []),
                metadata=analysis_data.get('metadata', {}),
                output_path=str(report_path)
            )
        
        # Clean up temp file
        if temp_file_path.exists():
            os.remove(temp_file_path)
        
        return {
            "success": True,
            "report_id": report_filename,
            "download_url": f"/api/download-report/{report_filename}"
        }
    
    except Exception as e:
        # Clean up temp file on error
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/api/download-report/{report_id}")
async def download_report(report_id: str):
    """Download generated PDF report."""
    report_path = REPORTS_DIR / report_id
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=str(report_path),
        filename=report_id,
        media_type="application/pdf"
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "PDF Analyzer API is running"}

# ---------------------------
# Enhanced Gradio Interface with Download Feature
# ---------------------------
def create_interface():
    custom_css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .output-tabs .tab-nav {
        background: #f5f5f5;
    }
    .download-section {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    """
    
    with gr.Blocks(css=custom_css, title="AI Research Paper Analyzer") as demo:
        with gr.Column(elem_classes=["main-container"]):
            # Header
            gr.HTML("""
            <div class="header">
                <h1>🤖 AI Research Paper Analyzer</h1>
                <p>Upload a PDF research paper to get comprehensive analysis with AI-generated figure explanations</p>
                <p><small>✨ New: Download PDF reports and use as API!</small></p>
            </div>
            """)
            
            # Input section
            with gr.Row():
                pdf_input = gr.File(
                    label="📄 Upload Research Paper (PDF)",
                    file_types=[".pdf"],
                    elem_id="pdf_upload"
                )
                with gr.Column():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Paper", 
                        variant="primary", 
                        size="lg",
                        visible=False
                    )
                    download_btn = gr.Button(
                        "📥 Generate & Download Report",
                        variant="secondary",
                        size="lg",
                        visible=False
                    )
            
            # Download section
            with gr.Row(visible=False, elem_classes=["download-section"]) as download_section:
                gr.HTML("""
                <div>
                    <h3>📋 Report Generated Successfully!</h3>
                    <p>Your comprehensive analysis report has been generated and is ready for download.</p>
                </div>
                """)
                download_file = gr.File(label="Download Report", visible=False)
            
            # Output section with tabs
            with gr.Tabs(visible=False) as output_tabs:
                with gr.Tab("📊 Text Analysis"):
                    text_analysis_output = gr.Textbox(
                        label="Comprehensive Paper Analysis",
                        lines=20,
                        max_lines=30
                    )
                
                with gr.Tab("🖼️ Figure Analysis"):
                    figures_output = gr.HTML(label="Extracted Figures with AI Captions")
                
                with gr.Tab("📋 Document Info"):
                    metadata_output = gr.Markdown(label="Document Metadata")
                
                with gr.Tab("🔗 API Usage"):
                    api_info = gr.HTML("""
                    <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
                        <h3>🚀 API Endpoints</h3>
                        <p>Use this analyzer as an API for your Next.js frontend:</p>
                        
                        <h4>1. Analyze PDF</h4>
                        <code>POST /api/analyze</code><br>
                        <small>Upload PDF file and get structured analysis data</small>
                        
                        <h4>2. Generate Report</h4>
                        <code>POST /api/generate-report</code><br>
                        <small>Generate and download PDF analysis report</small>
                        
                        <h4>3. Download Report</h4>
                        <code>GET /api/download-report/{report_id}</code><br>
                        <small>Download generated PDF report</small>
                        
                        <h4>4. Health Check</h4>
                        <code>GET /api/health</code><br>
                        <small>Check API status</small>
                        
                        <h4>Example Usage (Next.js):</h4>
                        <pre style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto;">
const analyzePDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/analyze', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  return data;
};

const downloadReport = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/generate-report', {
    method: 'POST',
    body: formData
  });
  
  const { download_url } = await response.json();
  window.open(download_url, '_blank');
};
                        </pre>
                    </div>
                    """)
            
            # Event handlers
            def show_buttons(file):
                if file:
                    return gr.update(visible=True), gr.update(visible=True)
                else:
                    return gr.update(visible=False), gr.update(visible=False)
            
            def run_comprehensive_analysis(pdf_file):
                if not pdf_file:
                    return None, None, None, gr.update(visible=False)
                
                text_analysis, figures_html, metadata_summary, analysis_data = analyze_pdf_comprehensive(pdf_file, gr.Progress())
                
                return (
                    gr.update(value=text_analysis),
                    gr.update(value=figures_html), 
                    gr.update(value=metadata_summary),
                    gr.update(visible=True)
                )
            
            def generate_and_download_report(pdf_file):
                if not pdf_file:
                    return None, gr.update(visible=False)
                
                try:
                    # Analyze PDF
                    text_analysis, figures_html, metadata_summary, analysis_data = analyze_pdf_comprehensive(pdf_file)
                    
                    # Generate report
                    timestamp = int(time.time())
                    report_filename = f"analysis_report_{timestamp}.pdf"
                    report_path = REPORTS_DIR / report_filename
                    
                    if analysis_data:
                        generate_pdf_report(
                            text_analysis=text_analysis,
                            image_infos=analysis_data.get('image_infos', []),
                            captions=analysis_data.get('captions', []),
                            analyses=analysis_data.get('analyses', []),
                            metadata=analysis_data.get('metadata', {}),
                            output_path=str(report_path)
                        )
                    
                    return str(report_path), gr.update(visible=True)
                    
                except Exception as e:
                    gr.Warning(f"Error generating report: {e}")
                    return None, gr.update(visible=False)
            
            pdf_input.change(
                show_buttons,
                inputs=[pdf_input],
                outputs=[analyze_btn, download_btn]
            )
            
            analyze_btn.click(
                run_comprehensive_analysis,
                inputs=[pdf_input],
                outputs=[text_analysis_output, figures_output, metadata_output, output_tabs],
                show_progress=True
            )
            
            download_btn.click(
                generate_and_download_report,
                inputs=[pdf_input],
                outputs=[download_file, download_section],
                show_progress=True
            )
    
    return demo

# ---------------------------
# Cleanup function
# ---------------------------
def cleanup_temp_files():
    """Clean up temporary files on startup."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(exist_ok=True)
    
    # Clean old reports (older than 1 hour)
    if REPORTS_DIR.exists():
        current_time = time.time()
        for report_file in REPORTS_DIR.glob("*.pdf"):
            if current_time - report_file.stat().st_mtime > 3600:  # 1 hour
                report_file.unlink()

# ---------------------------
# Run both Gradio and FastAPI
# ---------------------------
def run_fastapi():
    """Run FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_gradio():
    """Run Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

# ---------------------------
# Launch both servers
# ---------------------------
if __name__ == "__main__":
    cleanup_temp_files()
    
    # Run FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    print("🚀 FastAPI server started on http://localhost:8000")
    print("📋 API documentation available at http://localhost:8000/docs")
    print("🤖 Starting Gradio interface on http://localhost:7860")
    
    # Run Gradio in main thread
    run_gradio()