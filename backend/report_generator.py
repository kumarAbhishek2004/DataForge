"""
=========================================================
Analytics Copilot AI
PDF Report Generator
=========================================================
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class ReportGenerator:

    def __init__(self, save_path="reports/business_report.pdf"):
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.doc = SimpleDocTemplate(self.save_path, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.elements = []
        
        # Custom Title Style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor("#1e3a8a")
        )

    def add_title(self, title_text):
        self.elements.append(Paragraph(title_text, self.title_style))
        self.elements.append(Spacer(1, 12))

    def add_heading(self, heading_text):
        self.elements.append(Paragraph(heading_text, self.styles['Heading2']))
        self.elements.append(Spacer(1, 12))

    def add_paragraph(self, text):
        self.elements.append(Paragraph(text, self.styles['Normal']))
        self.elements.append(Spacer(1, 12))

    def add_image(self, image_path, width=400, height=250):
        if os.path.exists(image_path):
            img = Image(image_path, width=width, height=height)
            self.elements.append(img)
            self.elements.append(Spacer(1, 12))

    def add_table(self, dataframe):
        # Convert DataFrame to list of lists for ReportLab
        data = [dataframe.columns[:,].values.astype(str).tolist()] + dataframe.values.tolist()
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f3f4f6")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.elements.append(t)
        self.elements.append(Spacer(1, 20))

    def generate(self, insights_text=None, leaderboard_df=None, shap_image_path=None):
        """Compiles and builds the PDF."""
        self.add_title("Analytics Copilot AI: Business Report")
        
        self.add_heading("Executive Summary & AI Insights")
        if insights_text:
            self.add_paragraph(insights_text)
        else:
            self.add_paragraph("Model training complete. Review the sections below for detailed analytics.")
            
        if leaderboard_df is not None:
            self.add_heading("Model Leaderboard")
            self.add_table(leaderboard_df.head(5)) # Show top 5
            
        if shap_image_path and os.path.exists(shap_image_path):
            self.add_heading("Feature Importance & Explainability (SHAP)")
            self.add_image(shap_image_path)
            
        # Build the PDF
        self.doc.build(self.elements)
        return self.save_path
