
# ProductPricer-FineTuning-FrontierLLMs-QLoRA

This project aims to predict product prices using a combination of **traditional machine learning models**, **Frontier LLMs (Claude and GPT)**, and fine-tuning GPT models specifically for the task using **QLoRA**.

---

## **Project Overview**

The repository contains three major approaches to solving the product price prediction task:
1. **Traditional Machine Learning Models**: Linear regression, Random Forest, Gradient Boosting, and more.
2. **Frontier LLMs (Claude and GPT)**: Using state-of-the-art large language models to predict product prices.
3. **Fine-Tuning GPT**: Customizing GPT for product price prediction with **QLoRA** (Quantized LoRA).

---

## **Project Structure**

The repository is organized as follows:

```
FINE-TUNING FRONTIER LARGE LANGUAGE MODELS WITH LORAQLORA/
│
├── helpers/
│   ├── items.py          # Product data structures
│   ├── loaders.py        # Data loading utilities
│   ├── testing.py        # Testing utilities
│
├── .env                          # API keys for OpenAI, Claude, and Hugging Face
├── fine_tune_train.jsonl         # Training data for fine-tuning GPT
├── fine_tune_validation.jsonl    # Validation data for fine-tuning GPT
├── train.pkl                     # Preprocessed training dataset
├── test.pkl                      # Preprocessed testing dataset
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

---

## **Installation**

### **Clone the Repository**
```bash
git clone https://github.com/hamzabaccouri/ProductPricer-FineTuning-FrontierLLMs-QLoRA.git
cd ProductPricer-FineTuning-FrontierLLMs-QLoRA
```

### **Set Up Dependencies**
1. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # For Linux/Mac
   venv\Scripts\activate       # For Windows
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Set Up API Keys**
1. Open the `.env` file and provide your API keys:
   ```plaintext
   OPENAI_API_KEY=your_openai_key
   CLAUDE_API_KEY=your_claude_key
   HUGGINGFACE_API_KEY=your_huggingface_key
   ```

---

## **Usage**

### **1. Data Collection and Preprocessing**
- Use the data provided in `train.pkl` and `test.pkl` to train and evaluate the models.

### **2. Training Traditional Machine Learning Models**
- Utilize the helpers in the `helpers` folder to train traditional machine learning models like:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor

### **3. Frontier LLM Predictions**
- Use the Frontier LLMs (Claude and GPT) with data from `fine_tune_train.jsonl` and `fine_tune_validation.jsonl`.

### **4. Fine-Tuning GPT**
- Fine-tune GPT using the `fine_tune_train.jsonl` and `fine_tune_validation.jsonl` files with QLoRA to customize the model for product price prediction.

---

## **Features**

- **Data Preprocessing**: Clean and prepare data for price prediction tasks.
- **Traditional ML Models**: Establish baseline models using classical machine learning techniques.
- **Frontier LLMs**: Utilize state-of-the-art models (Claude and GPT) for price prediction.
- **Fine-Tuning**: Customize GPT-4O-mini using QLoRA for improved task-specific performance.

---
