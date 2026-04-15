"""
Generate a PDF explaining every section of the Streamlit Demo output page.
Creates annotated mockup images of each UI section and explains what it shows and why.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fpdf import FPDF
import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUTPUT_DIR, "_demo_images")
os.makedirs(IMG_DIR, exist_ok=True)


# =============================================================================
# GENERATE MOCKUP IMAGES OF EACH STREAMLIT SECTION
# =============================================================================

def img(name):
    return os.path.join(IMG_DIR, name)


def make_header_image():
    """Section 1: The page header and title."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    ax.text(5, 1.7, "\U0001F50D  Sentiment Analysis Bias Detector",
            fontsize=20, fontweight="bold", color="white",
            ha="center", va="center")
    ax.text(5, 0.9, "Test how AI sentiment analysis treats different names differently",
            fontsize=12, color="#888888", ha="center", va="center")
    ax.axhline(y=0.4, color="#333333", linewidth=1)
    ax.axis("off")
    fig.savefig(img("01_header.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_sidebar_image():
    """Section 2: Sidebar with pre-loaded examples."""
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.set_facecolor("#262730")
    fig.patch.set_facecolor("#262730")

    ax.text(2, 5.6, "\U0001F4CB Pre-loaded Examples", fontsize=13,
            fontweight="bold", color="white", ha="center")
    ax.text(2, 5.2, "Click to load an example:", fontsize=9, color="#aaa", ha="center")

    examples = [
        "\u25CB  Black Male - Angry",
        "\u25CB  White Male - Angry",
        "\u25CB  Black Female - Frustrated",
        "\u25CB  White Female - Frustrated",
        "\u25CB  Indian Male - Demanding",
        "\u25CB  Chinese Female - Disappointed",
    ]
    for i, ex in enumerate(examples):
        y = 4.6 - i * 0.6
        color = "#FF6B6B" if i == 0 else "#cccccc"
        if i == 0:
            ex = "\u25C9  Black Male - Angry"
        rect = mpatches.FancyBboxPatch((0.2, y - 0.2), 3.6, 0.45,
                                        boxstyle="round,pad=0.05",
                                        facecolor="#333340" if i == 0 else "#262730",
                                        edgecolor="#444", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.4, y, ex, fontsize=9, color=color, va="center")

    ax.axis("off")
    fig.savefig(img("02_sidebar.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_input_section_image():
    """Section 3: Input area - name + complaint + info box."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 2]})
    for ax in axes:
        ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    # Left: inputs
    ax = axes[0]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.text(0.2, 3.6, "\U0001F464 Customer Name", fontsize=10, color="#ccc")
    rect1 = mpatches.FancyBboxPatch((0.1, 2.9), 3.7, 0.55,
                                     boxstyle="round,pad=0.08",
                                     facecolor="#1A1A2E", edgecolor="#555")
    ax.add_patch(rect1)
    ax.text(0.3, 3.15, "Jamal Williams", fontsize=10, color="white")

    ax.text(0.2, 2.5, "\U0001F4AC Complaint Text", fontsize=10, color="#ccc")
    rect2 = mpatches.FancyBboxPatch((0.1, 0.8), 3.7, 1.5,
                                     boxstyle="round,pad=0.08",
                                     facecolor="#1A1A2E", edgecolor="#555")
    ax.add_patch(rect2)
    ax.text(0.3, 1.9, "I'm extremely angry about", fontsize=9, color="white")
    ax.text(0.3, 1.5, "my delayed package. This", fontsize=9, color="white")
    ax.text(0.3, 1.1, "is the third time!", fontsize=9, color="white")

    btn = mpatches.FancyBboxPatch((0.1, 0.1), 3.7, 0.5,
                                   boxstyle="round,pad=0.08",
                                   facecolor="#FF4B4B", edgecolor="#FF4B4B")
    ax.add_patch(btn)
    ax.text(2, 0.35, "\U0001F50D Analyze Sentiment", fontsize=10,
            color="white", ha="center", fontweight="bold")
    ax.axis("off")

    # Right: info box
    ax = axes[1]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    info_rect = mpatches.FancyBboxPatch((0.2, 0.3), 7.5, 3.4,
                                         boxstyle="round,pad=0.15",
                                         facecolor="#1B3A5C", edgecolor="#2980b9",
                                         linewidth=1.5)
    ax.add_patch(info_rect)
    ax.text(0.6, 3.3, "How this works:", fontsize=11, color="white", fontweight="bold")
    lines = [
        "1. Enter a customer name and complaint text",
        "2. The app analyzes sentiment using VADER",
        "3. It shows how a biased vs fair system would score",
        "4. The explanation reveals which words drive the prediction",
        "",
        "Try the pre-loaded examples to see bias in action!",
    ]
    for i, line in enumerate(lines):
        ax.text(0.6, 2.8 - i * 0.42, line, fontsize=9, color="#ddd")
    ax.axis("off")

    fig.savefig(img("03_input.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_demographic_detected_image():
    """Section 4: Detected demographic signal."""
    fig, ax = plt.subplots(figsize=(10, 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.axhline(y=1.0, color="#333", linewidth=1)
    ax.text(0.3, 0.5, "Detected Demographic Signal:  Black Male",
            fontsize=12, color="white", fontweight="bold")
    ax.axis("off")
    fig.savefig(img("04_demographic.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_results_image():
    """Section 5: Side-by-side biased vs fair results."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#0E1117")

    # BIASED column
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.set_facecolor("#0E1117")
    # Red header
    header_rect = mpatches.FancyBboxPatch((0.1, 5.0), 4.8, 0.85,
                                           boxstyle="round,pad=0.1",
                                           facecolor="#FFCDD2", edgecolor="#F44336",
                                           linewidth=2)
    ax.add_patch(header_rect)
    ax.text(2.5, 5.4, "\u26A0\uFE0F BIASED AI (Before Mitigation)",
            fontsize=11, color="#C62828", ha="center", fontweight="bold")

    ax.text(0.3, 4.5, "Sentiment Score", fontsize=10, color="#888")
    ax.text(0.3, 4.0, "-0.849", fontsize=22, color="white", fontweight="bold")

    ax.text(0.3, 3.3, "Severity:  Very Negative \U0001F534", fontsize=10, color="white")
    ax.text(0.3, 2.8, "Urgency Level:  CRITICAL", fontsize=10, color="white")
    ax.text(0.3, 2.3, "Est. Response Time:  1-2 hours", fontsize=10, color="white")

    err_rect = mpatches.FancyBboxPatch((0.1, 1.4), 4.8, 0.6,
                                        boxstyle="round,pad=0.08",
                                        facecolor="#5C1A1A", edgecolor="#FF4B4B",
                                        linewidth=1)
    ax.add_patch(err_rect)
    ax.text(2.5, 1.7, "Name bias applied: -0.150", fontsize=10,
            color="#FF6B6B", ha="center")

    ax.axis("off")

    # FAIR column
    ax = axes[1]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.set_facecolor("#0E1117")
    # Green header
    header_rect = mpatches.FancyBboxPatch((0.1, 5.0), 4.8, 0.85,
                                           boxstyle="round,pad=0.1",
                                           facecolor="#C8E6C9", edgecolor="#4CAF50",
                                           linewidth=2)
    ax.add_patch(header_rect)
    ax.text(2.5, 5.4, "\u2705 FAIR AI (After Mitigation)",
            fontsize=11, color="#2E7D32", ha="center", fontweight="bold")

    ax.text(0.3, 4.5, "Sentiment Score", fontsize=10, color="#888")
    ax.text(0.3, 4.0, "-0.699", fontsize=22, color="white", fontweight="bold")

    ax.text(0.3, 3.3, "Severity:  Very Negative \U0001F534", fontsize=10, color="white")
    ax.text(0.3, 2.8, "Urgency Level:  CRITICAL", fontsize=10, color="white")
    ax.text(0.3, 2.3, "Est. Response Time:  1-2 hours", fontsize=10, color="white")

    suc_rect = mpatches.FancyBboxPatch((0.1, 1.4), 4.8, 0.6,
                                        boxstyle="round,pad=0.08",
                                        facecolor="#1A3D1A", edgecolor="#4CAF50",
                                        linewidth=1)
    ax.add_patch(suc_rect)
    ax.text(2.5, 1.7, "No name bias applied  \u2713", fontsize=10,
            color="#66BB6A", ha="center")

    ax.axis("off")

    fig.suptitle("\U0001F4CA Results", fontsize=16, color="white", fontweight="bold", y=1.02)
    fig.savefig(img("05_results.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_bias_alert_image():
    """Section 6: The orange bias alert banner."""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    alert_rect = mpatches.FancyBboxPatch((0.2, 0.2), 9.6, 1.6,
                                          boxstyle="round,pad=0.15",
                                          facecolor="#FFF3E0", edgecolor="#FF9800",
                                          linewidth=2)
    ax.add_patch(alert_rect)
    ax.text(5, 1.35, "\u26A0\uFE0F BIAS DETECTED: 21% score difference due to name alone!",
            fontsize=13, color="#E65100", ha="center", fontweight="bold")
    ax.text(5, 0.75, "The biased system scored this complaint 15% more negatively",
            fontsize=10, color="#BF360C", ha="center")
    ax.text(5, 0.45, "because of the customer's name.",
            fontsize=10, color="#BF360C", ha="center")
    ax.axis("off")
    fig.savefig(img("06_bias_alert.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_explanation_image():
    """Section 7: LIME-style word explanation."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    ax.text(0.3, 4.6, "\U0001F52C Explanation (LIME-style)", fontsize=14,
            color="white", fontweight="bold")
    ax.text(0.3, 4.1, "Words pushing sentiment NEGATIVE:", fontsize=11,
            color="#ccc", fontweight="bold")

    words_data = [
        ("angry", -0.45, 18),
        ("delayed", -0.15, 6),
    ]
    for i, (word, weight, bar_len) in enumerate(words_data):
        y = 3.4 - i * 0.8
        ax.text(0.5, y, f"\u2022  {word}  \u2192  {weight:+.2f}",
                fontsize=11, color="white", family="monospace")
        bar_rect = mpatches.FancyBboxPatch((4.5, y - 0.15), bar_len * 0.25, 0.3,
                                            boxstyle="round,pad=0.03",
                                            facecolor="#FF4B4B", edgecolor="none")
        ax.add_patch(bar_rect)

    ax.text(0.3, 1.5, "\u26A0\uFE0F  Name effect:  Jamal Williams  adds  -0.150  bias",
            fontsize=11, color="#FFB74D", fontweight="bold")
    ax.text(0.3, 0.9, "If name were a White male name, biased score would be:",
            fontsize=10, color="#aaa")
    ax.text(0.3, 0.5, "-0.699  instead of  -0.849",
            fontsize=11, color="#FF8A80", fontweight="bold")

    ax.axis("off")
    fig.savefig(img("07_explanation.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_counterfactual_image():
    """Section 8: Counterfactual table."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")

    ax.text(0.3, 4.7, "\U0001F504 Counterfactual: Same Complaint, Different Names",
            fontsize=14, color="white", fontweight="bold")

    headers = ["Name", "Demographic", "Biased Score", "Fair Score", "Bias Applied", "Difference"]
    data = [
        ["Brad Johnson", "White Male", "-0.699", "-0.699", "+0.000", "0%"],
        ["Emily Wilson", "White Female", "-0.679", "-0.699", "+0.020", "2%"],
        ["Jamal Williams", "Black Male", "-0.849", "-0.699", "-0.150", "15%"],
        ["Lakisha Brown", "Black Female", "-0.819", "-0.699", "-0.120", "12%"],
        ["Amit Sharma", "Indian Male", "-0.769", "-0.699", "-0.070", "7%"],
        ["Wei Chen", "Chinese Male", "-0.759", "-0.699", "-0.060", "6%"],
    ]

    col_x = [0.2, 2.0, 3.6, 5.1, 6.5, 8.2]
    # Headers
    for j, h in enumerate(headers):
        ax.text(col_x[j], 4.1, h, fontsize=9, color="#AAD4FF", fontweight="bold")

    for i, row in enumerate(data):
        y = 3.5 - i * 0.55
        bg_color = "#1A1A2E" if i % 2 == 0 else "#14142A"
        rect = mpatches.Rectangle((0.1, y - 0.2), 9.7, 0.5,
                                   facecolor=bg_color, edgecolor="#333", linewidth=0.5)
        ax.add_patch(rect)
        for j, val in enumerate(row):
            color = "white"
            if j == 4 and val.startswith("-"):
                color = "#FF6B6B"
            elif j == 4 and val != "+0.000":
                color = "#66BB6A"
            ax.text(col_x[j], y, val, fontsize=9, color=color, va="center")

    ax.axis("off")
    fig.savefig(img("08_counterfactual.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def make_footer_image():
    """Section 9: Footer."""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.axhline(y=1.3, color="#333", linewidth=1)
    ax.text(5, 0.95, "Auditing Gender & Race Bias in Customer Service AI",
            fontsize=10, color="#888", ha="center", fontweight="bold")
    ax.text(5, 0.6, "Responsible AI Course Project | Demonstrating bias in sentiment analysis",
            fontsize=9, color="#666", ha="center")
    ax.text(5, 0.3, "This demo simulates documented bias patterns found in research.",
            fontsize=9, color="#666", ha="center")
    ax.axis("off")
    fig.savefig(img("09_footer.png"), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# =============================================================================
# BUILD THE PDF
# =============================================================================

class DemoPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("ArialUni", "", r"C:\Windows\Fonts\arial.ttf")
        self.add_font("ArialUni", "B", r"C:\Windows\Fonts\arialbd.ttf")
        self.add_font("ArialUni", "I", r"C:\Windows\Fonts\ariali.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("ArialUni", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Streamlit Demo Output - Explained", align="C")
            self.ln(3)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUni", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("ArialUni", "B", 16)
        self.set_text_color(25, 60, 120)
        self.ln(4)
        self.cell(0, 10, title)
        self.ln(6)
        self.set_draw_color(25, 60, 120)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def sub_title(self, title):
        self.set_font("ArialUni", "B", 13)
        self.set_text_color(40, 90, 160)
        self.ln(3)
        self.cell(0, 8, title)
        self.ln(6)

    def body(self, text):
        self.set_font("ArialUni", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("ArialUni", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_body(self, label, value):
        self.set_font("ArialUni", "B", 10)
        self.set_text_color(40, 40, 40)
        w = self.get_string_width(label) + 1
        self.cell(w, 5.5, label)
        self.set_font("ArialUni", "", 10)
        self.multi_cell(0, 5.5, value)
        self.ln(1)

    def highlight_box(self, text, color="blue"):
        colors = {
            "blue": (230, 240, 255, 25, 60, 120),
            "green": (230, 255, 230, 30, 120, 30),
            "red": (255, 230, 230, 180, 30, 30),
            "yellow": (255, 250, 220, 160, 120, 0),
        }
        bg_r, bg_g, bg_b, txt_r, txt_g, txt_b = colors.get(color, colors["blue"])
        self.set_fill_color(bg_r, bg_g, bg_b)
        self.set_text_color(txt_r, txt_g, txt_b)
        self.set_font("ArialUni", "B", 10)
        y = self.get_y()
        self.rect(10, y, 190, 12, "F")
        self.set_xy(14, y + 2)
        self.multi_cell(182, 5.5, text)
        self.ln(5)
        self.set_text_color(40, 40, 40)

    def add_img(self, filename, w=170):
        path = img(filename)
        if os.path.exists(path):
            if self.get_y() + 60 > 270:
                self.add_page()
            x = (210 - w) / 2
            self.image(path, x=x, w=w)
            self.ln(5)


def build_pdf():
    # Generate all images first
    print("Generating mockup images...")
    make_header_image()
    make_sidebar_image()
    make_input_section_image()
    make_demographic_detected_image()
    make_results_image()
    make_bias_alert_image()
    make_explanation_image()
    make_counterfactual_image()
    make_footer_image()
    print("All images generated.")

    pdf = DemoPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("ArialUni", "B", 26)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 14, "Streamlit Demo Output", align="C")
    pdf.ln(12)
    pdf.cell(0, 14, "Explained Section by Section", align="C")
    pdf.ln(16)

    pdf.set_draw_color(25, 60, 120)
    pdf.set_line_width(1)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(12)

    pdf.set_font("ArialUni", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Sentiment Analysis Bias Detector", align="C")
    pdf.ln(8)
    pdf.cell(0, 10, "What every element on the screen means and why it is shown", align="C")
    pdf.ln(25)

    pdf.set_font("ArialUni", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Example Used:  Jamal Williams  +  \"I'm extremely angry about my delayed package\"", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "This document walks through each section of the Streamlit output page", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "with annotated images and detailed explanations.", align="C")

    # =========================================================================
    # SECTION 1: HEADER
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 1: Page Header")
    pdf.add_img("01_header.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "At the very top of the page, there is a large title: "
        "\"Sentiment Analysis Bias Detector\" with a magnifying glass icon. "
        "Below it is a subtitle: \"Test how AI sentiment analysis treats different "
        "names differently.\" A horizontal line separates the header from the rest."
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "This header immediately tells you the PURPOSE of the app. It is a tool "
        "for testing and exposing bias. The subtitle makes it clear that the focus "
        "is on how NAMES (which signal race and gender) affect AI scores. "
        "A user who opens this page instantly knows what they are about to do."
    )

    # =========================================================================
    # SECTION 2: SIDEBAR
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 2: Sidebar - Pre-loaded Examples")
    pdf.add_img("02_sidebar.png", w=80)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "On the LEFT side of the screen, there is a collapsible sidebar titled "
        "\"Pre-loaded Examples\" with 6 radio button options:"
    )
    pdf.bullet("Black Male - Angry (Jamal Williams + angry complaint)")
    pdf.bullet("White Male - Angry (Brad Johnson + same angry complaint)")
    pdf.bullet("Black Female - Frustrated (Lakisha Brown + frustrated complaint)")
    pdf.bullet("White Female - Frustrated (Emily Wilson + same frustrated complaint)")
    pdf.bullet("Indian Male - Demanding (Rajesh Kumar + demanding complaint)")
    pdf.bullet("Chinese Female - Disappointed (Mei Chen + disappointed complaint)")

    pdf.sub_title("Why it is shown")
    pdf.body(
        "These examples are carefully PAIRED. Notice that \"Black Male - Angry\" "
        "and \"White Male - Angry\" use the EXACT SAME complaint text but different "
        "names. This lets you quickly compare: click one, see the result, click "
        "the other, see how the score changes. The pairing proves that "
        "any score difference is caused ONLY by the name."
    )
    pdf.ln(2)
    pdf.highlight_box(
        "TIP: The fastest way to see bias in action is to click \"Black Male - Angry\" "
        "first, then switch to \"White Male - Angry\" and compare.", "blue"
    )

    # =========================================================================
    # SECTION 3: INPUT AREA
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 3: Input Area")
    pdf.add_img("03_input.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see (Left Side)")
    pdf.body("The input area has three elements:")
    pdf.ln(2)

    pdf.bold_body("1. Customer Name field: ",
                  "A text input box where you type a name like \"Jamal Williams\" or "
                  "\"Brad Johnson\". The app uses this name to detect the demographic "
                  "group (race + gender) using a built-in name dictionary of 40+ names.")
    pdf.bold_body("2. Complaint Text area: ",
                  "A larger text box where you type the customer complaint. This is "
                  "the actual text that VADER analyzes for sentiment. You can type "
                  "anything or use the pre-loaded examples.")
    pdf.bold_body("3. Analyze Sentiment button: ",
                  "A red button that triggers the analysis. When clicked, the app "
                  "runs VADER on the complaint text, applies the bias simulation, "
                  "and displays all results below.")

    pdf.sub_title("What you see (Right Side)")
    pdf.body(
        "A blue information box explains how the app works in 4 simple steps. "
        "This guides new users who have never used the tool before."
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "The input area is where the experiment starts. By separating the NAME "
        "from the COMPLAINT TEXT, the app makes it visually clear that these are "
        "two different inputs. The name affects the BIASED score, while the "
        "complaint text determines the BASE score. Only by entering both can "
        "you see the bias in action."
    )

    # =========================================================================
    # SECTION 4: DETECTED DEMOGRAPHIC
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 4: Detected Demographic Signal")
    pdf.add_img("04_demographic.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "A single bold line appears after you click \"Analyze\": "
        "\"Detected Demographic Signal: Black Male\" (or whatever group the name "
        "belongs to). If the name is not in the database, it shows "
        "\"Not recognized (using neutral baseline)\"."
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "This is TRANSPARENCY. The app is telling you exactly what it detected "
        "from the name. It is saying: \"I recognized this name as belonging to a "
        "Black Male demographic group, and I will apply the corresponding bias "
        "factor.\" This makes the bias mechanism VISIBLE rather than hidden."
    )
    pdf.ln(2)
    pdf.sub_title("How the detection works")
    pdf.body(
        "The app has a NAME_DEMOGRAPHICS dictionary with 40+ name-to-demographic "
        "mappings. It first checks the full name (\"Jamal Williams\"), then tries "
        "just the first name (\"Jamal\"). These names come from published research "
        "(Bertrand & Mullainathan, 2004) that proved these names strongly signal "
        "a specific race and gender to other people and AI systems."
    )

    # =========================================================================
    # SECTION 5: RESULTS (BIASED vs FAIR)
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 5: Results - Biased vs Fair (The Core Output)")
    pdf.add_img("05_results.png", w=170)
    pdf.ln(2)

    pdf.body(
        "This is the MOST IMPORTANT section of the entire app. It shows two "
        "columns side by side, each representing a different AI system analyzing "
        "the SAME complaint."
    )

    pdf.sub_title("LEFT COLUMN: Biased AI (Red - Before Mitigation)")
    pdf.body("This column shows what happens when an AI system is BIASED - "
             "when it lets the customer's name influence the sentiment score.")
    pdf.ln(2)
    pdf.bold_body("Sentiment Score (-0.849): ",
                  "This is the VADER base score (-0.699) PLUS a name-based penalty "
                  "(-0.150 for Black Male). The formula is: Biased Score = Base Score "
                  "+ Bias Factor. The -0.150 penalty comes from the BIAS_FACTORS "
                  "dictionary which encodes the documented bias that Black male names "
                  "receive in real AI systems.")
    pdf.bold_body("Severity (Very Negative): ",
                  "Based on the biased score of -0.849. The app classifies scores: "
                  "below -0.6 = Very Negative (red circle), "
                  "-0.6 to -0.3 = Negative (orange), "
                  "-0.3 to 0 = Slightly Negative (yellow), "
                  "0+ = Neutral/Positive (green). "
                  "Because -0.849 is way below -0.6, it is classified as \"Very Negative\".")
    pdf.bold_body("Urgency Level (CRITICAL): ",
                  "In a real customer service system, the urgency level determines how "
                  "fast the team responds. CRITICAL means the system thinks this "
                  "customer is extremely upset and needs immediate attention. This is "
                  "derived directly from the sentiment score using the same thresholds.")
    pdf.bold_body("Est. Response Time (1-2 hours): ",
                  "This shows the practical IMPACT of the score. A biased score of "
                  "-0.849 triggers the fastest response time (1-2 hours). This means "
                  "the biased system would prioritize this customer as extremely urgent.")
    pdf.bold_body("Name bias applied: -0.150 (red alert): ",
                  "This red error box at the bottom EXPOSES the bias. It tells you "
                  "exactly how much the name changed the score. Without the name bias, "
                  "the score would be -0.699, but the name \"Jamal Williams\" pushed it "
                  "down by 0.150 points to -0.849.")

    pdf.add_page()
    pdf.sub_title("RIGHT COLUMN: Fair AI (Green - After Mitigation)")
    pdf.body("This column shows what happens when the AI is FAIR - when it ignores "
             "the customer's name and judges ONLY by the complaint text.")
    pdf.ln(2)
    pdf.bold_body("Sentiment Score (-0.699): ",
                  "This is the RAW VADER score with NO name adjustment. VADER reads "
                  "the words \"extremely angry\", \"delayed\", and calculates a compound "
                  "score of -0.699. This is the TRUE sentiment of the text.")
    pdf.bold_body("Severity (Very Negative): ",
                  "Even the fair score of -0.699 is below -0.6, so it is still "
                  "classified as \"Very Negative\". The complaint IS genuinely negative.")
    pdf.bold_body("Urgency Level (CRITICAL): ",
                  "Same classification as biased. The difference is: the fair system "
                  "reached this conclusion based on the WORDS, not the name.")
    pdf.bold_body("Est. Response Time (1-2 hours): ",
                  "Same response time. In this particular example, both systems happen "
                  "to give the same urgency. But for a borderline complaint (score near "
                  "-0.6), the name bias could push the biased system into a higher "
                  "urgency tier while the fair system stays in a lower tier.")
    pdf.bold_body("No name bias applied (green success box): ",
                  "This green box confirms that the fair system treated the customer "
                  "purely based on their words. Whether the customer is named Jamal, "
                  "Brad, Priya, or Wei - the fair score is always -0.699.")

    pdf.ln(4)
    pdf.sub_title("Why This Comparison Matters")
    pdf.body(
        "Side by side, you can see the EXACT impact of name-based bias. In this "
        "example, both columns happen to give the same severity, but the RAW SCORES "
        "differ by 0.150 points (21.5%). In other scenarios where the base score "
        "is closer to a threshold (like -0.59), the biased system might classify "
        "the complaint as \"Very Negative\" (CRITICAL, 1-2 hours) while the fair "
        "system correctly classifies it as just \"Negative\" (HIGH, 2-4 hours). "
        "That difference can mean a customer getting unnecessary escalation or "
        "being denied the support they need."
    )

    # =========================================================================
    # SECTION 6: BIAS ALERT BANNER
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 6: Bias Alert Banner")
    pdf.add_img("06_bias_alert.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "A prominent orange/yellow banner appears between the results and the "
        "explanation section. It shows a warning message like: "
        "\"BIAS DETECTED: 21% score difference due to name alone!\""
    )
    pdf.body(
        "Below the headline, it specifies: \"The biased system scored this complaint "
        "15% more negatively because of the customer's name.\""
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "This banner is designed to be IMPOSSIBLE TO MISS. Even if you don't "
        "understand sentiment scores or statistical metrics, this banner tells you "
        "in plain language exactly what happened."
    )
    pdf.ln(2)
    pdf.sub_title("How the numbers are calculated")
    pdf.bold_body("21% score difference: ",
                  "This is calculated as: |bias| / |fair_score| x 100 = 0.150 / 0.699 x 100 "
                  "= 21.5%. It represents how much the biased score deviates from the fair "
                  "score as a percentage.")
    pdf.bold_body("15% more negatively: ",
                  "This is the raw bias factor expressed as a percentage: "
                  "|bias| x 100 = 0.150 x 100 = 15%. It represents the absolute "
                  "penalty applied to the score.")
    pdf.ln(2)
    pdf.body(
        "This banner only appears when the name IS recognized and a non-zero bias "
        "is applied. If you enter a White Male name (bias = 0.00), no banner appears."
    )

    # =========================================================================
    # SECTION 7: LIME EXPLANATION
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 7: LIME-style Word Explanation")
    pdf.add_img("07_explanation.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "A section titled \"Explanation (LIME-style)\" showing which words in "
        "the complaint pushed the sentiment NEGATIVE. Each word is listed with "
        "its contribution score and a red bar showing its relative impact."
    )
    pdf.body("For the complaint \"I'm extremely angry about my delayed package\", you see:")
    pdf.bullet("angry  ->  -0.45  (long red bar) - the strongest negative word")
    pdf.bullet("delayed  ->  -0.15  (shorter red bar) - a milder negative word")
    pdf.ln(2)
    pdf.body(
        "Below the word list, if a name bias was detected, it shows: "
        "\"Name effect: Jamal Williams adds -0.150 bias in the biased system\" "
        "and \"If name were a White male name, biased score would be: -0.699 "
        "instead of -0.849\"."
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "This section answers the question: \"WHY did the AI give this score?\" "
        "It breaks down the prediction into individual word contributions so you "
        "can understand the reasoning. This is based on the concept of LIME "
        "(Local Interpretable Model-Agnostic Explanations), a real XAI technique."
    )
    pdf.ln(2)
    pdf.sub_title("How the scores work")
    pdf.body(
        "Each word has a pre-assigned emotion weight from a dictionary of 20 "
        "common complaint words. For example, \"angry\" = -0.45, \"furious\" = -0.55, "
        "\"frustrated\" = -0.35, \"disappointed\" = -0.25. The red bar length is "
        "proportional to the absolute value of the weight."
    )
    pdf.body(
        "The name effect line makes the bias EXPLICIT: it quantifies EXACTLY how "
        "much the name changed the score and what the score WOULD HAVE BEEN "
        "without the name bias."
    )

    # =========================================================================
    # SECTION 8: COUNTERFACTUAL TABLE
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 8: Counterfactual Table")
    pdf.add_img("08_counterfactual.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "A table titled \"Counterfactual: Same Complaint, Different Names\" showing "
        "6 rows. Each row takes the EXACT SAME complaint text you typed, but "
        "runs it with a different name. The columns are:"
    )
    pdf.ln(2)
    pdf.bold_body("Name: ", "The customer name used (Brad Johnson, Emily Wilson, Jamal Williams, etc.)")
    pdf.bold_body("Demographic: ", "The detected race + gender group")
    pdf.bold_body("Biased Score: ", "What the biased AI system would give this person")
    pdf.bold_body("Fair Score: ", "What the fair AI system gives (always the same: -0.699)")
    pdf.bold_body("Bias Applied: ", "The exact penalty or bonus applied to this demographic group")
    pdf.bold_body("Difference: ", "The bias as a percentage")
    pdf.ln(2)

    pdf.sub_title("The actual numbers")
    pdf.body("For the complaint \"I'm extremely angry about my delayed package\":")
    pdf.bullet("Brad Johnson (White Male): Biased = -0.699, Bias = +0.000 (ZERO bias, he is the baseline)")
    pdf.bullet("Emily Wilson (White Female): Biased = -0.679, Bias = +0.020 (POSITIVE bias, scored slightly less negatively)")
    pdf.bullet("Jamal Williams (Black Male): Biased = -0.849, Bias = -0.150 (scored 15% MORE negatively)")
    pdf.bullet("Lakisha Brown (Black Female): Biased = -0.819, Bias = -0.120 (scored 12% more negatively)")
    pdf.bullet("Amit Sharma (Indian Male): Biased = -0.769, Bias = -0.070 (scored 7% more negatively)")
    pdf.bullet("Wei Chen (Chinese Male): Biased = -0.759, Bias = -0.060 (scored 6% more negatively)")

    pdf.ln(2)
    pdf.sub_title("Why it is shown")
    pdf.body(
        "This is the SMOKING GUN of the demo. The complaint text never changed - "
        "only the name did. Yet the biased system gives 6 DIFFERENT scores. "
        "The fair system gives the SAME score (-0.699) every time."
    )
    pdf.ln(2)
    pdf.highlight_box(
        "This table proves the central claim of the project: AI bias causes the "
        "SAME complaint to be treated differently depending on the customer's name.", "red"
    )
    pdf.ln(2)
    pdf.body(
        "The Fair Score column being identical in every row is the proof of "
        "what mitigation achieves - equal treatment regardless of identity."
    )

    # =========================================================================
    # SECTION 9: FOOTER
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 9: Footer")
    pdf.add_img("09_footer.png", w=170)
    pdf.ln(2)

    pdf.sub_title("What you see")
    pdf.body(
        "At the bottom of the page, a small, centered footer shows the project title "
        "(\"Auditing Gender & Race Bias in Customer Service AI\"), identifies it as "
        "a \"Responsible AI Course Project\", and adds a disclaimer: \"This demo "
        "simulates documented bias patterns found in sentiment analysis research.\""
    )

    pdf.sub_title("Why it is shown")
    pdf.body(
        "The footer provides context and disclaimers. It clarifies that this is "
        "an educational/research tool and that the bias patterns shown are "
        "SIMULATED based on real research findings, not coming from a live "
        "biased AI model. This is important for academic integrity and to "
        "prevent misunderstanding."
    )

    # =========================================================================
    # SECTION 10: COMPLETE FLOW SUMMARY
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 10: Complete Flow Summary")

    pdf.body("Here is the complete flow of what happens when you use the demo:")
    pdf.ln(2)

    flow_steps = [
        ("Step 1 - You enter input: ",
         "Type a customer name (e.g., \"Jamal Williams\") and a complaint "
         "(e.g., \"I'm extremely angry about my delayed package\")"),
        ("Step 2 - Name detection: ",
         "The app looks up the name in its NAME_DEMOGRAPHICS dictionary and "
         "identifies the demographic group (Black Male)"),
        ("Step 3 - VADER analysis: ",
         "The app sends ONLY the complaint text (without the name) to VADER, "
         "which gives a base sentiment score of -0.699"),
        ("Step 4 - Biased score: ",
         "The app adds the BIAS_FACTORS penalty for Black Male (-0.15) to get "
         "the biased score: -0.699 + (-0.15) = -0.849"),
        ("Step 5 - Fair score: ",
         "The fair score is simply the base VADER score with NO name adjustment: -0.699"),
        ("Step 6 - Display results: ",
         "Both scores are shown side by side with severity, urgency, and "
         "response time classifications derived from score thresholds"),
        ("Step 7 - Bias alert: ",
         "If non-zero bias was applied, the orange banner shows the percentage "
         "difference: 21% score difference"),
        ("Step 8 - Explanation: ",
         "The LIME-style section breaks down which words drove the sentiment "
         "(angry = -0.45, delayed = -0.15) and quantifies the name's effect"),
        ("Step 9 - Counterfactual: ",
         "The same complaint is automatically run with 6 different names to "
         "show bias varies by demographic: 0% for White Male to 15% for Black Male"),
    ]
    for label, desc in flow_steps:
        pdf.bold_body(label, desc)
        pdf.ln(1)

    pdf.ln(4)
    pdf.highlight_box(
        "The entire demo is designed to make one message crystal clear: "
        "when AI is biased, the SAME complaint gets treated differently based "
        "on the customer's name. When AI is fair, everyone gets the same score.", "blue"
    )

    # =========================================================================
    # SECTION 11: BIAS FACTORS REFERENCE
    # =========================================================================
    pdf.add_page()
    pdf.section_title("Section 11: Bias Factors Reference")

    pdf.body(
        "The bias factors used in the demo are based on the actual findings from "
        "our BERT sentiment analysis experiments. Here are all 8 demographic "
        "groups and their bias penalties:"
    )
    pdf.ln(4)

    # Table
    headers = ["Demographic Group", "Bias Factor", "Effect", "Real-World Meaning"]
    data = [
        ["Black Male", "-0.150", "15% more negative", "Largest penalty - complaints scored most harshly"],
        ["Black Female", "-0.120", "12% more negative", "Second largest - compounding of race bias"],
        ["Indian Male", "-0.070", "7% more negative", "Moderate penalty for Indian names"],
        ["Chinese Male", "-0.060", "6% more negative", "Moderate penalty for Chinese names"],
        ["Indian Female", "-0.040", "4% more negative", "Smaller penalty - gender partially offsets race bias"],
        ["Chinese Female", "-0.030", "3% more negative", "Smallest minority penalty"],
        ["White Male", "+0.000", "No bias (baseline)", "Reference group - no adjustment applied"],
        ["White Female", "+0.020", "2% less negative", "Slight positive bias - complaints treated a bit milder"],
    ]

    col_widths = [38, 24, 36, 92]
    # Header row
    pdf.set_font("ArialUni", "B", 9)
    pdf.set_fill_color(25, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    # Data rows
    pdf.set_font("ArialUni", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    fill = False
    for row in data:
        if fill:
            pdf.set_fill_color(240, 245, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6.5, cell, border=1, fill=True, align="C")
        pdf.ln()
        fill = not fill

    pdf.ln(6)
    pdf.body(
        "These factors are derived from our BERT analysis which showed Black Male "
        "names getting 7.5% more negative scores than White Male names. The demo "
        "uses rounded, slightly amplified values (15% vs 7.5%) to make the bias "
        "more visually obvious in the demonstration, since the demo's purpose is "
        "educational - to make bias VISIBLE and understandable."
    )

    pdf.ln(4)
    pdf.sub_title("How the score ranges translate to actions")
    pdf.ln(2)
    headers2 = ["Score Range", "Severity", "Urgency", "Response Time", "Customer Impact"]
    data2 = [
        ["Below -0.6", "Very Negative", "CRITICAL", "1-2 hours", "Immediate escalation to manager"],
        ["-0.6 to -0.3", "Negative", "HIGH", "2-4 hours", "Priority queue, senior agent"],
        ["-0.3 to 0", "Slightly Negative", "MEDIUM", "4-8 hours", "Standard queue, regular agent"],
        ["0 and above", "Neutral/Positive", "LOW", "8+ hours", "Low priority, may auto-resolve"],
    ]
    col_widths2 = [28, 30, 26, 30, 76]
    pdf.set_font("ArialUni", "B", 9)
    pdf.set_fill_color(25, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers2):
        pdf.cell(col_widths2[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("ArialUni", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    fill = False
    for row in data2:
        if fill:
            pdf.set_fill_color(240, 245, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        for i, cell in enumerate(row):
            pdf.cell(col_widths2[i], 6.5, cell, border=1, fill=True, align="C")
        pdf.ln()
        fill = not fill

    pdf.ln(6)
    pdf.body(
        "When bias pushes a score from -0.55 (Negative) to -0.70 (Very Negative), "
        "it crosses the -0.6 threshold. This can change the urgency from HIGH to "
        "CRITICAL and the response time from 2-4 hours to 1-2 hours - a real, "
        "measurable impact on how the customer is treated."
    )

    # SAVE
    output_path = os.path.join(OUTPUT_DIR, "Demo_Output_Explained.pdf")
    pdf.output(output_path)
    print(f"\nPDF saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
