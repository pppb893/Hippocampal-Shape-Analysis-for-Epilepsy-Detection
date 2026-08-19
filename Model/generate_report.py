import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import KeepTogether

def parse_log(log_path):
    info = {
        'best_pls': 'N/A',
        'cv_acc': 'N/A',
        'test_acc': 'N/A',
        'precision_0': 'N/A',
        'recall_0': 'N/A',
        'f1_0': 'N/A',
        'precision_1': 'N/A',
        'recall_1': 'N/A',
        'f1_1': 'N/A'
    }
    if not os.path.exists(log_path):
        return info

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if 'Selected Best number of PLS components' in line:
            m = re.search(r'components:\s*(\d+)', line)
            if m:
                info['best_pls'] = m.group(1)
                
            m2 = re.search(r'with CV accuracy:\s*([0-9.]+)', line)
            if m2:
                info['cv_acc'] = m2.group(1)
            else:
                # If CV accuracy is not on the same line, try to find it from previous lines
                for prev_line in lines[:i]:
                    if f"PLS components: {info['best_pls']}," in prev_line:
                        m_prev = re.search(r'Accuracy.*?([0-9.]+)', prev_line)
                        if m_prev:
                            info['cv_acc'] = m_prev.group(1)
        if '*** Final Test Accuracy:' in line:
            m = re.search(r'Accuracy:\s*([0-9.]+)', line)
            if m:
                info['test_acc'] = m.group(1)
        if '0       ' in line and len(line.split()) >= 4 and info['precision_0'] == 'N/A':
            # Assumes format: 0       0.85      0.81      0.83        64
            parts = line.split()
            if len(parts) >= 4:
                info['precision_0'] = parts[1]
                info['recall_0'] = parts[2]
                info['f1_0'] = parts[3]
        if '1       ' in line and len(line.split()) >= 4 and info['precision_1'] == 'N/A':
            parts = line.split()
            if len(parts) >= 4:
                info['precision_1'] = parts[1]
                info['recall_1'] = parts[2]
                info['f1_1'] = parts[3]
                
    return info

def create_report():
    base_dir = r"c:\Users\jckky\OneDrive\Desktop\Hippocampal-Shape-Analysis-for-Epilepsy-Detection\Model"
    output_pdf = os.path.join(base_dir, "Model_Test_Report.pdf")
    
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    h1_style = styles['Heading1']
    h2_style = styles['Heading2']
    h3_style = styles['Heading3']
    normal_style = styles['Normal']
    
    # Custom styles
    h1_style.textColor = colors.HexColor("#2C3E50")
    h2_style.textColor = colors.HexColor("#2980B9")
    h3_style.textColor = colors.HexColor("#8E44AD")
    
    desc_style = ParagraphStyle(
        'DescStyle',
        parent=normal_style,
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("Hippocampal Shape Analysis - Model Evaluation Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("This document summarizes the performance and evaluation of various models across different dataset configurations.", desc_style))
    elements.append(Spacer(1, 0.5 * inch))
    
    # Model Descriptions
    elements.append(Paragraph("Model Code Descriptions", h1_style))
    models_info = {
        "MLP": "The Multi-Layer Perceptron (MLP) model first employs Partial Least Squares Discriminant Analysis (PLS-DA) with 5-fold cross-validation to select the optimal number of components and reduce feature dimensionality. The reduced features are then passed into a standard scikit-learn MLPClassifier for binary classification.",
        "MobileNet": "The MobileNet model utilizes a custom PyTorch 1D adaptation of MobileNetV2. It leverages inverted residual blocks and depthwise separable 1D convolutions to efficiently capture sequential/spatial patterns in the features, concluding with a global average pooling layer and linear classifier.",
        "ResNet": "The ResNet model is implemented as a custom 1D PyTorch architecture. It features standard residual blocks with skip connections and 1D convolutions, ensuring that gradient vanishing is mitigated while capturing hierarchical features from the hippocampal shape data.",
        "SVM": "The Support Vector Machine (SVM) pipeline works similarly to the MLP, first applying PLS-DA to find the best component count via cross-validation, and then fitting an SVM classifier (scikit-learn SVC) on the dimension-reduced dataset."
    }
    
    for model_name, desc in models_info.items():
        elements.append(Paragraph(model_name, h2_style))
        elements.append(Paragraph(desc, desc_style))
    
    elements.append(PageBreak())
    
    folders = [
        "All_Augment_tain",
        "Ds004469",
        "Ds004469Train_Ds005602test",
        "Ds005602",
        "Ds005602Train_Ds004469test"
    ]
    
    models = ["MLP", "MobileNet", "ResNet", "SVM"]
    sides = ["left", "right"]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        elements.append(Paragraph(f"Dataset Configuration: {folder}", h1_style))
        elements.append(Spacer(1, 0.1 * inch))
        
        for side in sides:
            elements.append(Paragraph(f"Hemisphere: {side.capitalize()}", h2_style))
            
            for model in models:
                log_file_name = f"{side}_{model}_train_{model.lower()}_pls.log"
                # Some files might have different names, fallback for ResNet
                if model == "ResNet":
                    # Check if standard resnet log exists, otherwise resnet_ae
                    std_log = os.path.join(folder_path, "logs", log_file_name)
                    if not os.path.exists(std_log):
                        log_file_name = f"{side}_{model}_train_resnet_ae_pls.log"
                        
                log_path = os.path.join(folder_path, "logs", log_file_name)
                
                info = parse_log(log_path)
                
                # Check if this model run exists
                plots_dir = os.path.join(folder_path, side, model, "plots")
                if not os.path.exists(log_path) and not os.path.exists(plots_dir):
                    continue
                    
                # Create a KeepTogether block for each model's summary to avoid breaking across pages awkwardly
                model_elements = []
                model_elements.append(Paragraph(f"Model: {model}", h3_style))
                
                # Create table for log data
                data = [
                    ["Metric", "Value"],
                    ["Test Accuracy", info['test_acc']],
                    ["Best PLS Components", info['best_pls']],
                    ["CV Accuracy (PLS)", info['cv_acc']],
                    ["Class 0 (Precision / Recall / F1)", f"{info['precision_0']} / {info['recall_0']} / {info['f1_0']}"],
                    ["Class 1 (Precision / Recall / F1)", f"{info['precision_1']} / {info['recall_1']} / {info['f1_1']}"]
                ]
                
                t = Table(data, colWidths=[2.5*inch, 3*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (1,0), colors.HexColor("#BDC3C7")),
                    ('TEXTCOLOR', (0,0), (1,0), colors.HexColor("#2C3E50")),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 6),
                    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#ECF0F1")),
                    ('GRID', (0,0), (-1,-1), 1, colors.white)
                ]))
                model_elements.append(t)
                model_elements.append(Spacer(1, 0.2 * inch))
                
                # Embed Plots
                images_to_embed = [
                    "roc_curve.png",
                    "confusion_matrix.png",
                    "roc_curve_ci.png",
                    "roc_curve_lines.png"
                ]
                
                img_elements = []
                for img_name in images_to_embed:
                    img_path = os.path.join(plots_dir, img_name)
                    if os.path.exists(img_path):
                        img = Image(img_path, width=2.5*inch, height=1.8*inch)
                        img_elements.append(img)
                        
                # Arrange images in pairs
                for i in range(0, len(img_elements), 2):
                    row = img_elements[i:i+2]
                    # Pad with empty spacer if odd number
                    if len(row) == 1:
                        row.append(Spacer(2.5*inch, 1.8*inch))
                    
                    # Add labels for images in paragraph format or simply put images in a table
                    img_table = Table([row], colWidths=[3*inch, 3*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
                    ]))
                    model_elements.append(img_table)
                    
                model_elements.append(Spacer(1, 0.3 * inch))
                elements.append(KeepTogether(model_elements))
                
        elements.append(PageBreak())

    doc.build(elements)
    print(f"PDF generated successfully at: {output_pdf}")

if __name__ == "__main__":
    create_report()
