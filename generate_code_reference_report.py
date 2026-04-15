"""
Generate a PDF report that explains every concept with exact
file paths and line numbers so the reader can jump to the code.
"""

from fpdf import FPDF
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class CodeRefPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("A", "", r"C:\Windows\Fonts\arial.ttf")
        self.add_font("A", "B", r"C:\Windows\Fonts\arialbd.ttf")
        self.add_font("A", "I", r"C:\Windows\Fonts\ariali.ttf")
        self.add_font("A", "BI", r"C:\Windows\Fonts\arialbi.ttf")

    def header(self):
        if self.page_no() > 1:
            self.set_font("A", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 7, "Code Reference Guide  |  Auditing Gender & Race Bias in Customer Service AI", align="C")
            self.ln(3)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("A", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, t):
        self.set_font("A", "B", 18)
        self.set_text_color(20, 55, 115)
        self.ln(4)
        self.cell(0, 11, t)
        self.ln(7)
        self.set_draw_color(20, 55, 115)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def h2(self, t):
        self.set_font("A", "B", 14)
        self.set_text_color(35, 85, 155)
        self.ln(3)
        self.cell(0, 9, t)
        self.ln(7)

    def h3(self, t):
        self.set_font("A", "B", 11)
        self.set_text_color(55, 55, 55)
        self.ln(2)
        self.cell(0, 7, t)
        self.ln(5)

    def p(self, t):
        self.set_font("A", "", 10)
        self.set_text_color(35, 35, 35)
        self.multi_cell(0, 5.5, t)
        self.ln(2)

    def bl(self, t, indent=10):
        self.set_font("A", "", 10)
        self.set_text_color(35, 35, 35)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, t)
        self.ln(1)

    def code_ref(self, file_path, line_start, line_end, description):
        """Print a styled code reference box."""
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(20, 55, 115)
        w = 190
        y = self.get_y()
        # Calculate needed height
        self.set_font("A", "B", 9)
        loc_text = f"File: {file_path}   |   Lines {line_start}-{line_end}"
        self.set_font("A", "", 9)
        # Draw box
        h = 16
        if self.get_y() + h > 265:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, w, h, "DF")
        # Location line
        self.set_xy(13, y + 2)
        self.set_font("A", "B", 9)
        self.set_text_color(20, 55, 115)
        self.cell(0, 5, loc_text)
        # Description line
        self.set_xy(13, y + 8)
        self.set_font("A", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, description)
        self.set_y(y + h + 2)
        self.set_text_color(35, 35, 35)

    def code_ref_tall(self, file_path, line_start, line_end, description, detail):
        """Code reference box with an extra detail line."""
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(20, 55, 115)
        w = 190
        y = self.get_y()
        h = 22
        if self.get_y() + h > 265:
            self.add_page()
            y = self.get_y()
        self.rect(10, y, w, h, "DF")
        self.set_xy(13, y + 2)
        self.set_font("A", "B", 9)
        self.set_text_color(20, 55, 115)
        self.cell(0, 5, f"File: {file_path}   |   Lines {line_start}-{line_end}")
        self.set_xy(13, y + 8)
        self.set_font("A", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, description)
        self.set_xy(13, y + 14)
        self.set_font("A", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(180, 4.5, detail)
        self.set_y(y + h + 2)
        self.set_text_color(35, 35, 35)

    def box(self, t, color="blue"):
        cols = {"blue": (225, 238, 255, 20, 55, 115),
                "green": (225, 250, 225, 25, 115, 25),
                "red": (255, 225, 225, 175, 25, 25),
                "yellow": (255, 248, 215, 155, 115, 0)}
        bg_r, bg_g, bg_b, t_r, t_g, t_b = cols.get(color, cols["blue"])
        self.set_fill_color(bg_r, bg_g, bg_b)
        self.set_text_color(t_r, t_g, t_b)
        self.set_font("A", "B", 10)
        y = self.get_y()
        self.rect(10, y, 190, 14, "F")
        self.set_xy(14, y + 3)
        self.multi_cell(182, 5.5, t)
        self.ln(5)
        self.set_text_color(35, 35, 35)

    def tbl(self, headers, data, cw=None):
        if cw is None:
            cw = [190 / len(headers)] * len(headers)
        self.set_font("A", "B", 9)
        self.set_fill_color(20, 55, 115)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(cw[i], 7, h, 1, 0, "C", True)
        self.ln()
        self.set_font("A", "", 8.5)
        self.set_text_color(35, 35, 35)
        fill = False
        for row in data:
            if self.get_y() > 262:
                self.add_page()
                self.set_font("A", "B", 9)
                self.set_fill_color(20, 55, 115)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(cw[i], 7, h, 1, 0, "C", True)
                self.ln()
                self.set_font("A", "", 8.5)
                self.set_text_color(35, 35, 35)
                fill = False
            self.set_fill_color(238, 243, 255) if fill else self.set_fill_color(255, 255, 255)
            for i, c in enumerate(row):
                self.cell(cw[i], 6.5, str(c), 1, 0, "C", True)
            self.ln()
            fill = not fill
        self.ln(4)

    def page_check(self, need=60):
        if self.get_y() + need > 268:
            self.add_page()


def build():
    pdf = CodeRefPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, 20)

    # =====================================================================
    # COVER PAGE
    # =====================================================================
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("A", "B", 28)
    pdf.set_text_color(20, 55, 115)
    pdf.cell(0, 14, "Code Reference Guide", align="C")
    pdf.ln(18)
    pdf.set_draw_color(20, 55, 115)
    pdf.set_line_width(1)
    pdf.line(45, pdf.get_y(), 165, pdf.get_y())
    pdf.ln(14)
    pdf.set_font("A", "B", 14)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Auditing Gender & Race Bias in Customer-Service AI", align="C")
    pdf.ln(12)
    pdf.set_font("A", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, "Every concept mapped to its exact file and line number", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "so you can navigate the codebase instantly.", align="C")
    pdf.ln(20)

    pdf.set_font("A", "B", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Project Files (All under  03_Code/ )", align="C")
    pdf.ln(10)
    pdf.set_font("A", "", 10)
    files_list = [
        "01_dataset_generation.py      -  Step 1: Dataset generation (800 sentences)",
        "02_sentiment_analysis.py      -  Step 2: VADER, TextBlob, BERT scoring",
        "03_bias_detection.py             -  Step 3: Statistical bias detection",
        "04_fairness_metrics.py          -  Step 4: 5 Fairlearn fairness metrics",
        "05_bias_mitigation.py            -  Step 5: 3 mitigation techniques",
        "06_explainability.py               -  Step 6: SHAP + LIME analysis",
        "07_privacy.py                         -  Step 7: Differential privacy (Opacus)",
        "08_visualizations.py              -  Step 8: 10 charts and plots",
        "run_all.py                                -  Master pipeline runner",
        "07_Demo/app.py                    -  Streamlit interactive demo",
    ]
    for f in files_list:
        pdf.cell(0, 7, f"       {f}", align="L")
        pdf.ln(7)

    # =====================================================================
    # FILE 1: 01_dataset_generation.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 1: 01_dataset_generation.py")
    pdf.p("This file creates the controlled 800-sentence complaint dataset. "
          "It is the foundation of the entire experiment.")

    pdf.h2("1.1 Complaint Templates (20 templates)")
    pdf.p("20 customer complaint templates across 4 emotion categories: "
          "angry (5), frustrated (5), disappointed (5), demanding (5).")
    pdf.code_ref("03_Code/01_dataset_generation.py", 27, 73,
                 "COMPLAINT_TEMPLATES dictionary - all 20 templates with emotion category and intensity")
    pdf.p("Each template uses {name} as a placeholder that gets replaced with a demographic name. "
          "Example: '{name} is angry about the delayed delivery'")
    pdf.bl("Lines 29-37: Angry templates (5) - 'angry', 'furious', 'outraged', 'upset', 'mad'")
    pdf.bl("Lines 40-48: Frustrated templates (5) - 'frustrated', 'annoyed', 'irritated', 'exasperated', 'fed up'")
    pdf.bl("Lines 51-59: Disappointed templates (5) - 'disappointed', 'let down', 'unsatisfied', 'cheated', 'unhappy'")
    pdf.bl("Lines 62-73: Demanding templates (5) - 'demands refund', 'speak to manager', 'threatens review', 'insists', 'expects'")

    pdf.page_check(50)
    pdf.h2("1.2 Demographic Names (8 groups x 5 names)")
    pdf.p("40 names from research by Bertrand & Mullainathan (2004) that strongly "
          "signal demographic identity.")
    pdf.code_ref("03_Code/01_dataset_generation.py", 79, 88,
                 "DEMOGRAPHIC_NAMES dictionary - 8 groups with 5 names each")
    pdf.bl("Line 80: Indian Male - Amit, Raj, Kumar, Aditya, Vikram")
    pdf.bl("Line 81: Indian Female - Priya, Ananya, Deepika, Kavya, Neha")
    pdf.bl("Line 82: White Male - Brad, Connor, Jake, Wyatt, Garrett")
    pdf.bl("Line 83: White Female - Emily, Molly, Katie, Megan, Allison")
    pdf.bl("Line 84: Black Male - DeShawn, Jamal, Darnell, Tyrone, Malik")
    pdf.bl("Line 85: Black Female - Lakisha, Latoya, Shaniqua, Tamika, Imani")
    pdf.bl("Line 86: Chinese Male - Wei, Ming, Chen, Zhang, Liu")
    pdf.bl("Line 87: Chinese Female - Ying, Mei, Xiu, Jing, Hui")

    pdf.page_check(50)
    pdf.h2("1.3 Dataset Generation Function")
    pdf.code_ref("03_Code/01_dataset_generation.py", 94, 122,
                 "generate_dataset() - Creates all 800 rows by combining templates x names")
    pdf.p("The function loops through every template (20), every demographic group (8), "
          "and every name (5), replacing {name} in the template text. Each row stores: "
          "Sentence_ID, Template_Number, Template_Category, Emotion_Intensity, Name, "
          "Demographic_Group, Race, Gender, Full_Text.")
    pdf.bl("Line 104: Nested loop - templates -> demographic groups -> names")
    pdf.bl("Line 106: race, gender = demo_group.rsplit('_', 1) extracts race and gender from group key")
    pdf.bl("Line 108: template_info['text'].format(name=name) replaces {name} placeholder")

    pdf.page_check(40)
    pdf.h2("1.4 Dataset Validation")
    pdf.code_ref("03_Code/01_dataset_generation.py", 125, 150,
                 "validate_dataset() - Checks completeness, balance by race, balance by gender")
    pdf.p("Validates that exactly 800 rows exist, 200 per race, 400 per gender. "
          "Ensures no sampling bias in the dataset itself.")

    # =====================================================================
    # FILE 2: 02_sentiment_analysis.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 2: 02_sentiment_analysis.py")
    pdf.p("Runs all 800 sentences through three sentiment analysis systems "
          "and records a numerical score for each.")

    pdf.h2("2.1 VADER Analysis")
    pdf.code_ref("03_Code/02_sentiment_analysis.py", 35, 50,
                 "analyze_vader() - Rule-based sentiment scoring using VADER dictionary")
    pdf.p("VADER uses a dictionary of ~7,500 words with pre-assigned sentiment scores. "
          "For each sentence, it looks up every word, adjusts for punctuation and "
          "capitalization (e.g., 'ANGRY!!!' scores more negative than 'angry'), "
          "then computes a compound score from -1 to +1.")
    pdf.bl("Line 43: SentimentIntensityAnalyzer() - creates the VADER engine")
    pdf.bl("Line 46: analyzer.polarity_scores(text) - returns dict with pos, neu, neg, compound")
    pdf.bl("Line 47-50: Stores compound, positive, neutral, negative sub-scores")
    pdf.box("WHY VADER HAS ZERO BIAS: It only checks the complaint words (angry, delayed, refund). "
            "Names like 'Jamal' or 'Brad' are NOT in VADER's dictionary, so they are ignored.", "green")

    pdf.page_check(60)
    pdf.h2("2.2 TextBlob Analysis")
    pdf.code_ref("03_Code/02_sentiment_analysis.py", 56, 70,
                 "analyze_textblob() - ML + Rules hybrid sentiment analysis")
    pdf.p("TextBlob uses a Naive Bayes classifier trained on movie reviews combined "
          "with pattern-based rules. Returns polarity (-1 to +1) and subjectivity (0 to 1).")
    pdf.bl("Line 65: TextBlob(text) - creates blob object for the sentence")
    pdf.bl("Line 67: blob.sentiment.polarity - extracts polarity score")
    pdf.bl("Line 68: blob.sentiment.subjectivity - measures objectivity vs opinion")
    pdf.box("WHY TEXTBLOB HAS ZERO BIAS: Its Naive Bayes model was trained on movie reviews, "
            "not on text with racial associations. Names have no learned sentiment weight.", "green")

    pdf.page_check(70)
    pdf.h2("2.3 BERT (DistilBERT) Analysis")
    pdf.code_ref("03_Code/02_sentiment_analysis.py", 76, 110,
                 "analyze_bert() - Deep learning transformer-based sentiment analysis")
    pdf.p("Uses HuggingFace's distilbert-base-uncased-finetuned-sst-2-english model. "
          "This is a 66-million parameter transformer that reads the ENTIRE sentence "
          "including the name as context.")
    pdf.bl("Line 87: pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')")
    pdf.bl("Line 88: device=-1 means CPU (change to 0 for GPU)")
    pdf.bl("Lines 93-101: Batch processing for efficiency (batch_size=16)")
    pdf.bl("Lines 96-101: Converts POSITIVE/NEGATIVE label + confidence into -1 to +1 scale")
    pdf.bl("Line 99: If label is NEGATIVE, score = -confidence (e.g., -0.95)")
    pdf.bl("Line 101: If label is POSITIVE, score = +confidence (e.g., +0.85)")
    pdf.box("WHY BERT IS BIASED: Pre-trained on Wikipedia + BookCorpus (internet text). "
            "It learned that names like 'Jamal' appear in more negative contexts than 'Brad'. "
            "So when it sees 'Jamal is angry...', the name adds negativity. Black Male: +7.5% more negative.", "red")

    pdf.page_check(50)
    pdf.h2("2.4 Combining All Results")
    pdf.code_ref("03_Code/02_sentiment_analysis.py", 116, 141,
                 "run_all_sentiment_analyses() - Orchestrates all 3 models and combines output")
    pdf.p("Calls analyze_vader(), analyze_textblob(), analyze_bert() in sequence, "
          "then concatenates all score columns with the original dataset into one DataFrame.")
    pdf.bl("Line 132: VADER scores   -> VADER_compound, VADER_positive, VADER_neutral, VADER_negative")
    pdf.bl("Line 136: TextBlob scores -> TextBlob_polarity, TextBlob_subjectivity")
    pdf.bl("Line 140: BERT scores    -> BERT_score, BERT_label, BERT_confidence")

    # =====================================================================
    # FILE 3: 03_bias_detection.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 3: 03_bias_detection.py")
    pdf.p("Contains all statistical tests to mathematically prove whether "
          "score differences between demographic groups are real or random noise.")

    pdf.h2("3.1 Bias Calculation Functions")
    pdf.code_ref("03_Code/03_bias_detection.py", 36, 56,
                 "calculate_bias_by_demographic() - Computes bias for each group vs White Male baseline")
    pdf.p("Groups all 800 sentences by Demographic_Group, calculates the mean score "
          "for each group, then subtracts the White_Male mean to get the bias value. "
          "Also calculates percentage bias.")
    pdf.bl("Line 40: group_stats = df.groupby('Demographic_Group')[score_col].agg(['mean', 'std', 'count'])")
    pdf.bl("Line 43: baseline_mean = group_stats.loc[baseline_group, 'mean'] (White_Male)")
    pdf.bl("Line 47: bias = group mean - baseline mean (positive = rated more positively)")
    pdf.bl("Line 48: pct_bias = (bias / |baseline_mean|) * 100")

    pdf.page_check(50)
    pdf.code_ref("03_Code/03_bias_detection.py", 59, 74,
                 "calculate_bias_by_race() - Same but grouped by race only (collapsing gender)")
    pdf.code_ref("03_Code/03_bias_detection.py", 77, 93,
                 "calculate_bias_by_category() - Bias broken down by emotion type (angry, frustrated, etc.)")

    pdf.page_check(70)
    pdf.h2("3.2 Statistical Significance Tests")
    pdf.code_ref("03_Code/03_bias_detection.py", 99, 146,
                 "run_statistical_tests() - t-test, Cohen's d, Mann-Whitney U for each group vs White Male")
    pdf.p("For each of the 7 non-baseline groups, this function computes:")
    pdf.bl("Line 109: t_stat, p_value = stats.ttest_ind(baseline_scores, group_scores)  -- Independent t-test")
    pdf.bl("Lines 112-114: Cohen's d = (mean1 - mean2) / pooled_std  -- Effect size measure")
    pdf.bl("Lines 117-119: stats.mannwhitneyu()  -- Non-parametric backup test (doesn't assume normality)")
    pdf.bl("Lines 123-135: Stores all results including: Difference, T_Statistic, P_Value, Cohens_d, Effect_Size_Label")
    pdf.bl("Line 130: Effect size labels: |d| >= 0.8 Large, >= 0.5 Medium, >= 0.2 Small, else Negligible")

    pdf.page_check(40)
    pdf.h2("3.3 ANOVA Test")
    pdf.code_ref("03_Code/03_bias_detection.py", 139, 143,
                 "run_anova_by_race() - One-way ANOVA to test if ANY race group differs")
    pdf.p("Uses scipy.stats.f_oneway() to test the null hypothesis that all race groups "
          "have the same mean sentiment score. If p < 0.05, at least one group is different.")

    pdf.h2("3.4 Intersectional Analysis")
    pdf.code_ref("03_Code/03_bias_detection.py", 155, 200,
                 "intersectional_analysis() - Analyzes how race x gender INTERACT to create compound bias")
    pdf.p("Creates a pivot table (Race rows x Gender columns) of mean scores. "
          "Calculates interaction effects to show that being Black AND Male creates "
          "compounding bias greater than either factor alone.")

    # =====================================================================
    # FILE 4: 04_fairness_metrics.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 4: 04_fairness_metrics.py")
    pdf.p("Applies 5 industry-standard fairness metrics using Microsoft's Fairlearn library.")

    pdf.h2("4.1 Data Preparation")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 41, 54,
                 "prepare_binary_labels() - Converts continuous scores to binary labels for fairness metrics")
    pdf.p("Fairness metrics need binary predictions (0 or 1). This function converts "
          "continuous scores into: predicted_label (positive if >= 0) and predicted_severity "
          "(high if below median).")
    pdf.bl("Line 48: true_label = 0 (all complaints are actually negative - ground truth)")
    pdf.bl("Line 49: predicted_label = (score >= threshold).astype(int)")
    pdf.bl("Line 52: predicted_severity based on median score")

    pdf.page_check(50)
    pdf.h2("4.2 Metric 1: Demographic Parity Difference")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 60, 69,
                 "demographic_parity_difference() - Are positive prediction rates equal across groups?")
    pdf.p("Uses fairlearn.metrics.demographic_parity_difference. Measures whether all "
          "racial groups get 'positive' predictions at the same rate. Ideal = 0, threshold < 0.10.")

    pdf.h2("4.3 Metric 4: Disparate Impact Ratio")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 72, 80,
                 "demographic_parity_ratio() - 80% rule: ratio of worst/best group positive rates")
    pdf.p("Legal standard from US Equal Employment Opportunity Commission. If the ratio "
          "of the lowest group's positive rate to the highest is below 0.80, it's a legal violation.")

    pdf.page_check(40)
    pdf.h2("4.4 Metric 3: Equalized Odds Difference")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 90, 98,
                 "equalized_odds_difference() - Equal TPR and FPR across groups?")
    pdf.p("Checks that both true positive rates AND false positive rates are equal across groups. "
          "Combines two measures into one: ideal = 0, threshold < 0.10.")

    pdf.h2("4.5 Metric 2: Equal Opportunity Difference")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 101, 117,
                 "equal_opportunity_difference_manual() - Equal true positive rates?")
    pdf.p("Manual implementation since all true labels are 0 (negative). Uses prediction "
          "rate as proxy. Returns the max-min difference across groups.")

    pdf.page_check(40)
    pdf.h2("4.6 Metric 5: Calibration Difference")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 120, 143,
                 "calibration_by_group() - Does prediction confidence mean the same across groups?")
    pdf.p("Calculates mean, std, median, Q25, Q75 of scores per group. The calibration "
          "difference is the max-min of group means. Ideal = 0, threshold < 0.10.")

    pdf.page_check(50)
    pdf.h2("4.7 Complete Fairness Report Generator")
    pdf.code_ref("03_Code/04_fairness_metrics.py", 149, 240,
                 "generate_fairness_report() - Runs all 5 metrics and prints pass/fail for one system")
    pdf.p("This is the master function that orchestrates all 5 metrics for a given "
          "sentiment system (VADER, TextBlob, or BERT). It:")
    pdf.bl("Line 162: Prepares binary labels from continuous scores")
    pdf.bl("Lines 170-178: Metric 1 - Demographic Parity Difference (using Fairlearn)")
    pdf.bl("Lines 187-192: Metric 2 - Equal Opportunity Difference (manual)")
    pdf.bl("Lines 196-203: Metric 3 - Equalized Odds Difference (using Fairlearn)")
    pdf.bl("Lines 207-214: Metric 4 - Disparate Impact Ratio (using Fairlearn)")
    pdf.bl("Lines 218-224: Metric 5 - Calibration Difference (custom)")
    pdf.bl("Lines 228-235: Overall verdict: X/5 PASSED -> Fair/Partially Biased/Severely Biased")

    # =====================================================================
    # FILE 5: 05_bias_mitigation.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 5: 05_bias_mitigation.py")
    pdf.p("Implements three bias mitigation strategies from pre-processing, "
          "in-processing, and post-processing stages of the ML pipeline.")

    pdf.h2("5.1 Data Preparation for Mitigation")
    pdf.code_ref("03_Code/05_bias_mitigation.py", 26, 47,
                 "prepare_model_data() - Creates TF-IDF features and binary labels from text")
    pdf.p("Trains a TF-IDF vectorizer (max 500 features) on the complaint text, "
          "creates binary labels from VADER median split, and extracts sensitive attributes.")
    pdf.bl("Line 36: TfidfVectorizer(max_features=500, stop_words='english')")
    pdf.bl("Line 37: X = vectorizer.fit_transform(df['Full_Text']) -> sparse matrix")
    pdf.bl("Lines 39-42: Extracts y (labels), sensitive_race, sensitive_gender, sensitive_demo as numpy arrays")

    pdf.page_check(60)
    pdf.h2("5.2 Baseline Model (Biased)")
    pdf.code_ref("03_Code/05_bias_mitigation.py", 53, 91,
                 "train_baseline_model() - Standard LogisticRegression with no fairness constraints")
    pdf.p("Trains a vanilla LogisticRegression as the reference point. Measures accuracy, "
          "demographic parity, equalized odds, and disparate impact ratio.")
    pdf.bl("Line 62: model = LogisticRegression(max_iter=1000, random_state=42)")
    pdf.bl("Line 63: model.fit(X_train, y_train) - standard training, no fairness")
    pdf.bl("Lines 68-71: Fairlearn metrics on the predictions")
    pdf.bl("Lines 74-78: Manual disparate impact calculation")

    pdf.page_check(70)
    pdf.h2("5.3 Mitigation 1: Reweighing (Pre-processing)")
    pdf.code_ref("03_Code/05_bias_mitigation.py", 97, 146,
                 "apply_reweighing() - Adjusts sample weights to balance group representation before training")
    pdf.p("Calculates a weight for each training sample based on its (group, label) combination. "
          "Underrepresented (group, label) pairs get higher weights.")
    pdf.bl("Lines 110-117: Weight calculation loop: weight = expected_count / actual_count")
    pdf.bl("Line 114: expected = (n_group * n_label) / n_total")
    pdf.bl("Line 115: weight = expected / n_group_label")
    pdf.bl("Line 119: model.fit(X_train, y_train, sample_weight=sample_weights)")
    pdf.p("Result: Had ZERO effect on our data because names have minimal predictive power in TF-IDF.")

    pdf.page_check(70)
    pdf.h2("5.4 Mitigation 2: Exponentiated Gradient (In-processing)")
    pdf.code_ref("03_Code/05_bias_mitigation.py", 152, 200,
                 "apply_exponentiated_gradient() - Fairlearn's constrained optimization during training")
    pdf.p("Uses Fairlearn's ExponentiatedGradient algorithm with a DemographicParity constraint. "
          "Trains multiple models iteratively, adjusting weights to satisfy fairness constraints.")
    pdf.bl("Line 165: base_model = LogisticRegression(max_iter=1000)")
    pdf.bl("Line 166: constraint = DemographicParity()  -- the fairness constraint")
    pdf.bl("Line 168: mitigator = ExponentiatedGradient(estimator=base_model, constraints=constraint, max_iter=50)")
    pdf.bl("Line 172: X_train_dense = X_train.toarray()  -- Fairlearn needs dense arrays")
    pdf.bl("Line 174: mitigator.fit(X_train_dense, y_train, sensitive_features=sensitive_train)")

    pdf.page_check(70)
    pdf.h2("5.5 Mitigation 3: Threshold Optimizer (Post-processing)")
    pdf.code_ref("03_Code/05_bias_mitigation.py", 207, 250,
                 "apply_threshold_optimizer() - Adjusts decision boundaries per group after training")
    pdf.p("Keeps the trained baseline model but finds optimal per-group thresholds to satisfy "
          "a demographic parity constraint.")
    pdf.bl("Line 220: postprocessor = ThresholdOptimizer(estimator=baseline_model, constraints='demographic_parity', prefit=True)")
    pdf.bl("Line 227: postprocessor.fit(X_train_dense, y_train, sensitive_features=sensitive_train)")
    pdf.bl("Line 229: y_pred = postprocessor.predict(X_test_dense, sensitive_features=sensitive_test)")
    pdf.p("Result: -1.7% accuracy, fairness slightly worse (0.136 vs 0.119). "
          "Shows the difficulty of post-hoc fixing.")

    # =====================================================================
    # FILE 6: 06_explainability.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 6: 06_explainability.py")
    pdf.p("Uses SHAP and LIME to explain WHY the model makes certain predictions "
          "and whether demographic names are driving the output.")

    pdf.h2("6.1 Model Preparation for Explainability")
    pdf.code_ref("03_Code/06_explainability.py", 25, 46,
                 "prepare_explainability_model() - Trains a TF-IDF + LogReg model specifically for XAI analysis")
    pdf.p("Creates a simpler model (200 TF-IDF features instead of 500) specifically "
          "for explainability analysis. Splits 70/30 train/test.")
    pdf.bl("Line 32: TfidfVectorizer(max_features=200, stop_words='english')")
    pdf.bl("Line 35: train_test_split with stratify=df['label'] for balanced splits")
    pdf.bl("Line 39: LogisticRegression(max_iter=1000, random_state=42)")

    pdf.page_check(70)
    pdf.h2("6.2 SHAP Analysis")
    pdf.code_ref("03_Code/06_explainability.py", 52, 96,
                 "run_shap_analysis() - Global + local SHAP explanations using LinearExplainer")
    pdf.p("Uses shap.LinearExplainer (optimized for linear models) to compute "
          "Shapley values for every feature in every prediction.")
    pdf.bl("Line 65: explainer = shap.LinearExplainer(model, X_train, feature_names=...)")
    pdf.bl("Line 68: shap_values = explainer.shap_values(X_test)")
    pdf.bl("Lines 72-79: Calculates mean absolute SHAP value per feature -> global importance ranking")
    pdf.bl("Lines 84-88: Saves shap_summary_plot.png - visual feature importance")
    pdf.bl("Lines 93-117: Compares SHAP values across demographics for emotion words")
    pdf.bl("Lines 120-141: Individual explanation examples - same template, different names")

    pdf.page_check(60)
    pdf.h2("6.3 SHAP Key Finding")
    pdf.box("SHAP proved that complaint words (refund: 0.164, product: 0.158) have 100x more "
            "influence than names (jamal: 0.0002). Names are NOT driving predictions in the "
            "TF-IDF model. BERT's bias comes from its contextual encoding, not word features.", "blue")

    pdf.page_check(70)
    pdf.h2("6.4 LIME Analysis")
    pdf.code_ref("03_Code/06_explainability.py", 150, 215,
                 "run_lime_analysis() - Individual prediction explanations using text perturbation")
    pdf.p("Creates localized explanations by perturbing individual sentences and "
          "observing prediction changes.")
    pdf.bl("Line 164: explainer = LimeTextExplainer(class_names=['Negative', 'Positive'], random_state=42)")
    pdf.bl("Lines 167-169: predict_fn = lambda texts: model.predict_proba(vectorizer.transform(texts))")
    pdf.bl("Lines 178-181: Comparison pairs - same emotion, different demographics")
    pdf.bl("Line 193: exp_a = explainer.explain_instance(text_a, predict_fn, num_features=10, num_samples=500)")
    pdf.bl("Lines 211-215: Saves interactive HTML files: lime_White_Male_angry.html, etc.")

    pdf.page_check(50)
    pdf.h2("6.5 LIME Counterfactual Analysis")
    pdf.code_ref("03_Code/06_explainability.py", 218, 260,
                 "Counterfactual - replaces names in the same sentence and compares predictions")
    pdf.p("Takes each base complaint (e.g., 'is angry about the delayed delivery'), "
          "prepends different demographic names, and compares prediction differences. "
          "This directly isolates the name's effect on the model output.")

    # =====================================================================
    # FILE 7: 07_privacy.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 7: 07_privacy.py")
    pdf.p("Implements differential privacy using two approaches: Laplace noise injection "
          "and Opacus DP-SGD for PyTorch neural networks.")

    pdf.h2("7.1 Laplace Noise Injection")
    pdf.code_ref("03_Code/07_privacy.py", 28, 32,
                 "add_laplace_noise() - Core DP function that adds calibrated random noise")
    pdf.p("Adds Laplace(0, sensitivity/epsilon) noise to any numerical data. "
          "Lower epsilon = more noise = more privacy = potentially less accuracy.")
    pdf.bl("Line 30: scale = sensitivity / epsilon")
    pdf.bl("Line 31: noise = np.random.laplace(0, scale, size=data.shape)")

    pdf.page_check(60)
    pdf.h2("7.2 Training with DP (Output Perturbation)")
    pdf.code_ref("03_Code/07_privacy.py", 35, 71,
                 "train_with_dp_noise() - Trains normally, then adds Laplace noise to model weights")
    pdf.p("This is 'output perturbation': train the model normally, then add calibrated noise "
          "to the learned coefficients before making predictions.")
    pdf.bl("Line 47: model = LogisticRegression(max_iter=1000)")
    pdf.bl("Line 48: model.fit(X_train, y_train)  -- normal training first")
    pdf.bl("Line 51: sensitivity = 2.0 / (len(y_train) * model.C)  -- bound on how much one sample affects coefficients")
    pdf.bl("Line 54: noisy_coef = add_laplace_noise(model.coef_, epsilon, sensitivity)")
    pdf.bl("Line 55-57: model.coef_ = noisy_coef  -- replace weights with noisy version")

    pdf.page_check(80)
    pdf.h2("7.3 DP-SGD with Opacus (Meta's Library)")
    pdf.code_ref("03_Code/07_privacy.py", 77, 148,
                 "train_with_opacus() - Gold standard DP for deep learning using per-sample gradient clipping + noise")
    pdf.p("Opacus modifies PyTorch's SGD to be differentially private:")
    pdf.bl("Lines 100-106: Defines a 2-layer neural network (Linear->ReLU->Linear)")
    pdf.bl("Line 107: ModuleValidator.fix(model) - validates model is compatible with Opacus")
    pdf.bl("Lines 112-118: privacy_engine.make_private_with_epsilon() - wraps model/optimizer/dataloader")
    pdf.bl("Line 115: target_epsilon=target_epsilon - the desired privacy budget")
    pdf.bl("Line 116: target_delta=1e-5 - probability of privacy failure")
    pdf.bl("Line 117: max_grad_norm=1.0 - per-sample gradient clipping bound")
    pdf.bl("Lines 121-127: Training loop - Opacus automatically clips gradients and adds noise each step")
    pdf.bl("Line 138: actual_epsilon = privacy_engine.get_epsilon(delta=1e-5) - reports actual privacy spent")

    pdf.page_check(60)
    pdf.h2("7.4 Epsilon Sensitivity Analysis")
    pdf.code_ref("03_Code/07_privacy.py", 154, 197,
                 "epsilon_sensitivity_analysis() - Tests 7 epsilon values from 0.1 to infinity")
    pdf.p("Tests how different privacy levels affect accuracy and fairness. "
          "For each epsilon, runs 5 times with different random seeds for stability.")
    pdf.bl("Line 163: epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]")
    pdf.bl("Lines 167-172: 5 runs per epsilon, averages accuracy and fairness")
    pdf.bl("Lines 174-180: Classification: Maximum/Very Strong/Strong/Moderate/Weak/None")

    pdf.page_check(60)
    pdf.h2("7.5 Three-Way Tradeoff Analysis")
    pdf.code_ref("03_Code/07_privacy.py", 203, 250,
                 "three_way_tradeoff() - Tests Accuracy vs Fairness vs Privacy together")
    pdf.p("Tests 4 configurations: (1) Baseline, (2) Fair only, (3) Private only, "
          "(4) Fair + Private. Shows that no single configuration achieves all three goals.")

    # =====================================================================
    # FILE 8: 08_visualizations.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 8: 08_visualizations.py")
    pdf.p("Generates 10 professional charts for the report using matplotlib and seaborn.")

    pdf.h2("8.1 Chart Configuration")
    pdf.code_ref("03_Code/08_visualizations.py", 17, 35,
                 "Global matplotlib settings - DPI, font sizes, color maps for demographics")
    pdf.bl("Lines 27-32: RACE_COLORS dictionary - Green (White), Red (Black), Blue (Indian), Orange (Chinese)")
    pdf.bl("Lines 34-41: DEMO_COLORS - 8 colors for all demographic groups")

    charts = [
        ("Chart 1: Bias by Demographic", "47, 86",
         "chart_bias_by_demographic() - 3-panel bar chart showing bias for each group across all 3 systems"),
        ("Chart 2: Bias by Race", "92, 133",
         "chart_bias_by_race() - Grouped bar chart comparing VADER/TextBlob/BERT bias by race"),
        ("Chart 3: Fairness Heatmap", "139, 200",
         "chart_fairness_heatmap() - Pass/fail heatmap for 5 metrics x 3 systems"),
        ("Chart 4: Intersectionality", "206, 247",
         "chart_intersectionality() - Race x Gender heatmap showing compounding effects"),
        ("Chart 5: Box Plots", "253, 290",
         "chart_box_plots() - Score distribution box plots by demographic"),
        ("Chart 6: Bias by Emotion", "296, 336",
         "chart_bias_by_emotion() - How bias varies across angry/frustrated/disappointed/demanding"),
        ("Chart 7: Mitigation Comparison", "342, 390",
         "chart_mitigation_comparison() - Before/after for 3 mitigation techniques"),
        ("Chart 8: Privacy-Accuracy", "395, 435",
         "chart_privacy_accuracy() - Accuracy vs epsilon across 7 privacy levels"),
        ("Chart 9: Responsible AI Stack", "440, 490",
         "chart_responsible_ai_stack() - Complete fairness+explainability+privacy framework"),
        ("Chart 10: Impact Projection", "495, 540",
         "chart_impact_projection() - Real-world business impact at scale"),
    ]

    for title, lines, desc in charts:
        pdf.page_check(30)
        pdf.h3(title)
        pdf.code_ref("03_Code/08_visualizations.py", lines.split(", ")[0], lines.split(", ")[1], desc)

    # =====================================================================
    # FILE 9: run_all.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 9: run_all.py (Master Runner)")
    pdf.p("Orchestrates the entire 8-step pipeline end-to-end.")

    pdf.h2("9.1 Module Loading")
    pdf.code_ref("03_Code/run_all.py", 32, 35,
                 "load_module() - Uses importlib to import files with numeric prefixes (01_, 02_, etc.)")

    pdf.h2("9.2 Pipeline Steps")
    steps = [
        ("Step 1: Generate Dataset", "38, 46", "step1_generate_dataset() - Calls 01_dataset_generation.generate_dataset() and saves CSV"),
        ("Step 2: Sentiment Analysis", "49, 82", "step2_sentiment_analysis() - Loads dataset, runs VADER/TextBlob/BERT, saves scored CSV"),
        ("Step 3: Bias Detection", "85, 95", "step3_bias_detection() - Loads scored data, runs all statistical tests, saves results"),
        ("Step 4: Fairness Metrics", "97, 107", "step4_fairness_metrics() - Runs 5 Fairlearn metrics on all 3 systems"),
        ("Step 5: Bias Mitigation", "109, 119", "step5_bias_mitigation() - Applies 3 mitigation techniques, compares results"),
        ("Step 6: Explainability", "121, 131", "step6_explainability() - Runs SHAP and LIME analysis"),
        ("Step 7: Privacy", "133, 143", "step7_privacy() - Epsilon sensitivity + Opacus DP-SGD"),
        ("Step 8: Visualizations", "145, 155", "step8_visualizations() - Generates all 10 charts"),
    ]
    for title, lines, desc in steps:
        pdf.page_check(25)
        pdf.h3(title)
        pdf.code_ref("03_Code/run_all.py", lines.split(", ")[0], lines.split(", ")[1], desc)

    # =====================================================================
    # FILE 10: 07_Demo/app.py
    # =====================================================================
    pdf.add_page()
    pdf.h1("File 10: 07_Demo/app.py (Streamlit Demo)")
    pdf.p("Interactive web application that demonstrates bias in sentiment "
          "analysis with side-by-side biased vs fair comparison.")

    pdf.h2("10.1 Model Loaders")
    pdf.code_ref("07_Demo/app.py", 28, 47,
                 "load_vader(), load_textblob(), load_bert_pipeline() - Cached model loading with @st.cache_resource")
    pdf.bl("Line 30: @st.cache_resource ensures VADER loads only once across Streamlit reruns")
    pdf.bl("Line 36: @st.cache_resource for TextBlob")
    pdf.bl("Lines 41-47: load_bert_pipeline() - loads DistilBERT with try/except fallback")

    pdf.page_check(50)
    pdf.h2("10.2 Score Functions")
    pdf.code_ref("07_Demo/app.py", 50, 62,
                 "get_vader_score(), get_textblob_score(), get_bert_score() - Wrappers for each model")
    pdf.bl("Line 51: get_vader_score() - returns analyzer.polarity_scores(text)['compound']")
    pdf.bl("Line 56: get_textblob_score() - returns TextBlob(text).sentiment.polarity")
    pdf.bl("Line 61: get_bert_score() - runs pipeline, converts to -1/+1 scale")

    pdf.page_check(60)
    pdf.h2("10.3 Bias Simulation System")
    pdf.p("The demo uses VADER for the base score (because VADER is unbiased) and "
          "then SIMULATES what a biased system would do by adding research-based bias factors.")
    pdf.code_ref("07_Demo/app.py", 68, 104,
                 "NAME_DEMOGRAPHICS - Maps 48 names to their (Race, Gender) tuple")
    pdf.code_ref("07_Demo/app.py", 107, 116,
                 "BIAS_FACTORS - Simulated bias offsets based on BERT research findings")
    pdf.bl("Line 108: ('Black', 'Male'): -0.15   -- Most biased (based on BERT's 7.5% finding)")
    pdf.bl("Line 109: ('Black', 'Female'): -0.12")
    pdf.bl("Line 110: ('Indian', 'Male'): -0.07")
    pdf.bl("Line 113: ('White', 'Male'): 0.00    -- Baseline (no bias)")
    pdf.bl("Line 114: ('White', 'Female'): 0.02")

    pdf.page_check(50)
    pdf.h2("10.4 Why VADER Instead of BERT in the Demo?")
    pdf.box("ANSWER: The demo needs a CLEAN unbiased base score to show the DIFFERENCE "
            "between fair and biased. VADER gives the TRUE sentiment. Then we ADD simulated "
            "bias to show what BERT would do. If we used BERT directly, both sides would "
            "already be biased and we couldn't show the contrast.", "blue")
    pdf.p("The flow is:")
    pdf.bl("Step 1: Get VADER score of the complaint text (unbiased, name-independent)")
    pdf.bl("Step 2: FAIR AI side: Display the VADER score as-is (no bias)")
    pdf.bl("Step 3: BIASED AI side: Take VADER score + BIAS_FACTORS[demographic] (simulates BERT)")
    pdf.bl("Step 4: Show the difference = the bias caused by the name alone")

    pdf.code_ref("07_Demo/app.py", 119, 133,
                 "detect_demographic(), simulate_biased_score(), simulate_fair_score() - Core bias simulation")
    pdf.bl("Line 125: simulate_biased_score(): returns np.clip(base_score + bias, -1, 1)")
    pdf.bl("Line 130: simulate_fair_score(): simply returns base_score unchanged")

    pdf.page_check(60)
    pdf.h2("10.5 LIME-Style Explanation Generator")
    pdf.code_ref("07_Demo/app.py", 139, 163,
                 "generate_explanation() - Creates word-level contribution breakdown for the prediction")
    pdf.p("Uses a hardcoded emotion word dictionary (20 words with sentiment weights) "
          "to show which words push the score negative. Simpler than full LIME but gives "
          "the same visual effect in the demo.")
    pdf.bl("Lines 145-156: emotion_words dictionary - 'angry': -0.45, 'furious': -0.55, etc.")
    pdf.bl("Lines 158-162: Loops through sentence words, matches against dictionary")

    pdf.page_check(60)
    pdf.h2("10.6 Main UI - Analysis Flow")
    pdf.code_ref("07_Demo/app.py", 250, 265,
                 "Analysis button handler - Gets VADER base score, applies biased/fair simulation")
    pdf.bl("Line 257: base_score = get_vader_score(vader, complaint_text)")
    pdf.bl("Lines 260-262: biased_score, race, gender, bias = simulate_biased_score(base_score, customer_name)")
    pdf.bl("Line 263: fair_score = simulate_fair_score(base_score)")

    pdf.page_check(50)
    pdf.h2("10.7 Results Display")
    pdf.code_ref("07_Demo/app.py", 275, 345,
                 "Side-by-side columns: Red BIASED panel (left) vs Green FAIR panel (right)")
    pdf.bl("Lines 280-286: Biased panel - red background with warning icon")
    pdf.bl("Lines 288-295: Severity calculation from score thresholds (-0.6, -0.3, 0)")
    pdf.bl("Lines 312-318: Fair panel - green background with checkmark")

    pdf.page_check(50)
    pdf.h2("10.8 Counterfactual Table")
    pdf.code_ref("07_Demo/app.py", 370, 400,
                 "Counterfactual analysis - Same complaint, 6 different names, shows bias for each")
    pdf.p("Loops through 6 test names, applies simulate_biased_score() to each, "
          "displays a DataFrame showing Biased Score, Fair Score, Bias Applied, and % Difference.")

    # =====================================================================
    # OUTPUT FILES REFERENCE
    # =====================================================================
    pdf.add_page()
    pdf.h1("Output Files Reference")
    pdf.p("All generated output files and where the code that creates them lives.")

    pdf.tbl(
        ["Output File", "Created By", "Lines"],
        [
            ["02_Data/complaint_dataset_800.csv", "01_dataset_generation.py", "94-122"],
            ["02_Data/sentiment_scores_all_systems.csv", "02_sentiment_analysis.py", "116-141"],
            ["04_Results/bias_by_demographic_VADER.csv", "03_bias_detection.py", "36-56"],
            ["04_Results/bias_by_demographic_BERT.csv", "03_bias_detection.py", "36-56"],
            ["04_Results/statistical_tests_BERT.csv", "03_bias_detection.py", "99-146"],
            ["04_Results/fairness_metrics_all_systems.csv", "04_fairness_metrics.py", "149-240"],
            ["04_Results/mitigation_comparison.csv", "05_bias_mitigation.py", "53-250"],
            ["04_Results/shap_feature_importance.csv", "06_explainability.py", "80-83"],
            ["04_Results/shap_summary_plot.png", "06_explainability.py", "86-90"],
            ["04_Results/lime_*.html (7 files)", "06_explainability.py", "211-215"],
            ["04_Results/lime_comparison_results.csv", "06_explainability.py", "193-210"],
            ["04_Results/privacy_epsilon_analysis.csv", "07_privacy.py", "154-197"],
            ["04_Results/three_way_tradeoff.csv", "07_privacy.py", "203-250"],
            ["04_Results/chart1_bias_by_demographic.png", "08_visualizations.py", "47-86"],
            ["04_Results/chart2_bias_by_race.png", "08_visualizations.py", "92-133"],
            ["04_Results/chart3_fairness_heatmap.png", "08_visualizations.py", "139-200"],
            ["04_Results/chart4_intersectionality.png", "08_visualizations.py", "206-247"],
            ["04_Results/chart5_box_plots.png", "08_visualizations.py", "253-290"],
            ["04_Results/chart6_bias_by_emotion.png", "08_visualizations.py", "296-336"],
            ["04_Results/chart7_mitigation_comparison.png", "08_visualizations.py", "342-390"],
            ["04_Results/chart8_privacy_accuracy.png", "08_visualizations.py", "395-435"],
            ["04_Results/chart9_responsible_ai_stack.png", "08_visualizations.py", "440-490"],
            ["04_Results/chart10_impact_projection.png", "08_visualizations.py", "495-540"],
        ],
        [68, 58, 64]
    )

    # =====================================================================
    # QUICK LOOKUP TABLE
    # =====================================================================
    pdf.add_page()
    pdf.h1("Quick Lookup: Concept -> Code Location")
    pdf.p("Find any concept instantly by looking up its file and line number.")

    pdf.tbl(
        ["Concept / Feature", "File", "Line(s)"],
        [
            ["Complaint templates (20)", "01_dataset_generation.py", "27-73"],
            ["Demographic names (40)", "01_dataset_generation.py", "79-88"],
            ["Dataset generation loop", "01_dataset_generation.py", "94-122"],
            ["VADER sentiment scoring", "02_sentiment_analysis.py", "35-50"],
            ["TextBlob sentiment scoring", "02_sentiment_analysis.py", "56-70"],
            ["BERT/DistilBERT scoring", "02_sentiment_analysis.py", "76-110"],
            ["BERT model name & config", "02_sentiment_analysis.py", "87-88"],
            ["Score combination (-1/+1)", "02_sentiment_analysis.py", "96-101"],
            ["Bias calculation formula", "03_bias_detection.py", "36-56"],
            ["Independent t-test", "03_bias_detection.py", "109"],
            ["Cohen's d effect size", "03_bias_detection.py", "112-114"],
            ["Mann-Whitney U test", "03_bias_detection.py", "117-119"],
            ["One-way ANOVA", "03_bias_detection.py", "139-143"],
            ["Intersectional analysis", "03_bias_detection.py", "155-200"],
            ["Binary label conversion", "04_fairness_metrics.py", "41-54"],
            ["Demographic Parity (Fairlearn)", "04_fairness_metrics.py", "60-69"],
            ["Disparate Impact Ratio", "04_fairness_metrics.py", "72-80"],
            ["Equalized Odds", "04_fairness_metrics.py", "90-98"],
            ["Equal Opportunity", "04_fairness_metrics.py", "101-117"],
            ["Calibration metric", "04_fairness_metrics.py", "120-143"],
            ["5-metric fairness report", "04_fairness_metrics.py", "149-240"],
            ["TF-IDF vectorizer setup", "05_bias_mitigation.py", "36-37"],
            ["Baseline LogisticRegression", "05_bias_mitigation.py", "62-63"],
            ["Reweighing (sample weights)", "05_bias_mitigation.py", "110-119"],
            ["ExponentiatedGradient", "05_bias_mitigation.py", "165-174"],
            ["ThresholdOptimizer", "05_bias_mitigation.py", "220-229"],
            ["SHAP LinearExplainer", "06_explainability.py", "65-68"],
            ["SHAP feature importance", "06_explainability.py", "72-83"],
            ["SHAP summary plot", "06_explainability.py", "86-90"],
            ["SHAP demographic comparison", "06_explainability.py", "93-117"],
            ["LIME text explainer", "06_explainability.py", "164-169"],
            ["LIME counterfactual pairs", "06_explainability.py", "178-181"],
            ["LIME HTML report generation", "06_explainability.py", "211-215"],
            ["Laplace noise function", "07_privacy.py", "28-32"],
            ["DP output perturbation", "07_privacy.py", "35-71"],
            ["Sensitivity calculation", "07_privacy.py", "51"],
            ["Opacus DP-SGD setup", "07_privacy.py", "100-118"],
            ["Per-sample gradient clipping", "07_privacy.py", "117"],
            ["Privacy budget tracking", "07_privacy.py", "138"],
            ["Epsilon sensitivity loop", "07_privacy.py", "163-180"],
            ["Three-way tradeoff configs", "07_privacy.py", "203-250"],
            ["Matplotlib style config", "08_visualizations.py", "17-41"],
            ["Demo: VADER base scoring", "07_Demo/app.py", "257"],
            ["Demo: Bias simulation factors", "07_Demo/app.py", "107-116"],
            ["Demo: Biased score = base + bias", "07_Demo/app.py", "125"],
            ["Demo: Counterfactual table", "07_Demo/app.py", "370-400"],
            ["Demo: LIME-style explanation", "07_Demo/app.py", "139-163"],
            ["Pipeline: run all 8 steps", "run_all.py", "38-155"],
        ],
        [56, 62, 72]
    )

    # =====================================================================
    # FINAL PAGE
    # =====================================================================
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("A", "B", 20)
    pdf.set_text_color(20, 55, 115)
    pdf.cell(0, 14, "End of Code Reference Guide", align="C")
    pdf.ln(18)
    pdf.set_draw_color(20, 55, 115)
    pdf.set_line_width(1)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("A", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Every function, every metric, every chart -- mapped to its exact code location.", align="C")
    pdf.ln(10)
    pdf.set_font("A", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Open any file, go to the line number, and see the implementation.", align="C")

    # SAVE
    out = os.path.join(OUTPUT_DIR, "Code_Reference_Guide.pdf")
    pdf.output(out)
    print(f"\nPDF saved to: {out}")


if __name__ == "__main__":
    build()
