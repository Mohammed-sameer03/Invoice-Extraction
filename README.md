The goal of this project is to automatically extract key information (like item name, price, tax, total, etc.) from invoice images and classify each extracted text segment into predefined categories using a fine-tuned Transformer model.

⚙️ What I Did:

Used EasyOCR to extract text from raw invoice images.

Collected and cleaned the extracted text into labeled fields (e.g., menu_item, tax_price, total_price).

Handled class imbalance and prepared a structured dataset.

Used DistilBERT, a pretrained Transformer model, and fine-tuned it for multi-class text classification.

Split data into training and testing sets to evaluate model performance.

Achieved high classification accuracy after fine-tuning on the domain-specific data.

Built an interactive Streamlit app to upload invoices, run OCR, and display predicted labels for each extracted text line.

🧠 Tech Stack:

Python, TensorFlow, Hugging Face Transformers

EasyOCR, scikit-learn

Streamlit (for user interface)
