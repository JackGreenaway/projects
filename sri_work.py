import docx2txt
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QFileDialog,
)
from PyQt5.QtCore import Qt


class FeedbackWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize NLTK resources
        nltk.download("vader_lexicon")
        nltk.download("punkt")
        nltk.download("cmudict")
        self.sia = SentimentIntensityAnalyzer()

        # Set up the window
        self.setGeometry(200, 200, 800, 500)
        self.setWindowTitle("Document Feedback")

        # Read the document file
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Word Document", "", "Word Documents (*.docx)"
        )
        document_text = docx2txt.process(file_path)

        # Compute the Flesch Reading Ease score
        sentences = sent_tokenize(document_text)
        flesch_score = sum(
            self.compute_flesch_reading_ease_score(
                sentence, self.sia.polarity_scores(sentence)["compound"]
            )
            for sentence in sentences
        ) / max(1, len(sentences))

        # Compute the sentiment scores
        positive_score = 0
        negative_score = 0
        neutral_score = 0
        for sentence in sentences:
            scores = self.sia.polarity_scores(sentence)
            if scores["compound"] >= 0.05:
                positive_score += 1
            elif scores["compound"] <= -0.05:
                negative_score += 1
            else:
                neutral_score += 1

        # Compute the marks
        knowledge_and_understanding_mark = (
            self.compute_knowledge_and_understanding_mark(flesch_score)
        )
        criticality_mark = self.compute_criticality_mark(
            positive_score, negative_score, neutral_score
        )
        reading_and_research_mark = self.compute_reading_and_research_mark(flesch_score)
        overall_mark = (
            knowledge_and_understanding_mark * 0.3
            + criticality_mark * 0.3
            + reading_and_research_mark * 0.3
            + round(
                (flesch_score + positive_score + neutral_score - negative_score) * 100
            )
            * 0.1
        )

        # Create the table data
        table_data = [
            [
                "Mark Range",
                "Knowledge & Understanding (30%)",
                "Criticality (30%)",
                "Reading and Research (30%)",
                "Writing Style (10%)",
            ],
            ["0%", "0", "0", "0", "0"],
            ["1%-39%", "0", "0", "0", "0"],
            ["40%-49%", "0", "0", "0", "0"],
            ["50%-59%", "0", "0", "0", "0"],
            ["60%-69%", "0", "0", "0", "0"],
            ["70%-79%", "0", "0", "0", "0"],
            ["80%-100%", "0", "0", "0", "0"],
        ]

        # Update the table data
        self.update_table_data(
            table_data,
            flesch_score,
            positive_score,
            negative_score,
            neutral_score,
            overall_mark,
        )

        # Create the table widget
        table_widget = QTableWidget(self)
        table_widget.setGeometry(50, 100, self.width() - 100, 250)
        table_widget.setRowCount(len(table_data))
        table_widget.setColumnCount(len(table_data[0]))
        table_widget.setHorizontalHeaderLabels(table_data[0])

        # Add the table data to
        for i, row in enumerate(table_data):
            for j, col in enumerate(row):
                table_item = QTableWidgetItem(str(col))
                table_item.setTextAlignment(Qt.AlignCenter)
                table_widget.setItem(i, j, table_item)

        # Set up the overall mark label
        overall_mark_label = QLabel(self)
        overall_mark_label.setText(f"Overall Mark: {overall_mark}%")
        overall_mark_label.setStyleSheet("font-size: 18px;")
        overall_mark_label.adjustSize()
        overall_mark_label.move(
            self.width() // 2 - overall_mark_label.width() // 2, 380
        )

        self.show()

    def update_table_data(
        self,
        table_data,
        flesch_score,
        positive_score,
        negative_score,
        neutral_score,
        overall_mark,
    ):
        # Update the Writing Style column
        flesch_row = ""
        if flesch_score >= 90:
            flesch_row = "80%-100%"
        elif flesch_score >= 80:
            flesch_row = "70%-79%"
        elif flesch_score >= 70:
            flesch_row = "60%-69%"
        elif flesch_score >= 60:
            flesch_row = "50%-59%"
        elif flesch_score >= 50:
            flesch_row = "40%-49%"
        elif flesch_score >= 1:
            flesch_row = "1%-39%"
        else:
            flesch_row = "0%"

        for row in table_data[1:]:
            if row[0] == flesch_row:
                row[4] = round(
                    (flesch_score + positive_score + neutral_score - negative_score)
                    * 100
                )

        # Update the last row
        table_data[-1][1] = round(table_data[-1][1] * 0.3)
        table_data[-1][2] = round(table_data[-1][2] * 0.3)
        table_data[-1][3] = round(table_data[-1][3] * 0.3)
        table_data[-1][4] = round(table_data[-1][4] * 0.1)
        for row in table_data[1:]:
            table_data[-1][1] += row[1]
            table_data[-1][2] += row[2]
            table_data[-1][3] += row[3]
            table_data[-1][4] += row[4]
        table_data[-1][1] = str(table_data[-1][1]) + "%"
        table_data[-1][2] = str(table_data[-1][2]) + "%"
        table_data[-1][3] = str(table_data[-1][3]) + "%"
        table_data[-1][4] = str(table_data[-1][4]) + "%"

    def compute_flesch_reading_ease_score(self, sentence, sentiment_score):
        # Compute the Flesch Reading Ease score for a sentence
        word = word_tokenize(sentence)
        syllables = sum(
            len(word) > 2
            and word[-2:] == "es"
            or word[-1] == "e"
            and word[-2] not in "aeiouy"
            or len(word) > 1
            and word[-1] == "e"
            and word[-2] not in "aeiouy"
            or len(word) > 1
            and word[-1] == "y"
            and word[-2] not in "aeiouy"
        )
        if sum(word.lower().count(vowel) for vowel in "aeiou"):
            for words in word:
                return (
                    206.835
                    - 1.015 * len(words) / len(sentence)
                    - 84.6 * syllables / len(words)
                )

    def compute_sentiment_scores(self, sentence):
        # Compute the sentiment scores for a sentence
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(sentence)
        return ss["pos"], ss["neg"], ss["neu"]

        if name == "main":
            app = QApplication(sys.argv)
            feedback_window = FeedbackWindow()
            sys.exit(app.exec_())


FeedbackWindow()
