
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
ProductPricer-FineTuning-FrontierLLMs-QLoRA/
│
├── helpers/
│   ├── items.py          # Product data structures
│   ├── loaders.py        # Data loading utilities
│   ├── testing.py        # Testing utilities
│
├── notebooks/
│   ├── dataCollection.ipynb               # Collect and preprocess data
│   ├── ProductPricerBaseModel.ipynb       # Train and evaluate traditional ML models
│   ├── ProductPricerFrontierLLM.ipynb     # Predict prices using Claude and GPT
│   ├── FineTuneOpenAIProductPricer.ipynb  # Fine-tune GPT for price prediction
│
├── data/
│   ├── train.pkl                 # Preprocessed training dataset
│   ├── test.pkl                  # Preprocessed test dataset
│   ├── fine_tune_train.jsonl     # Training data for fine-tuning GPT
│   ├── fine_tune_validation.jsonl # Validation data for fine-tuning GPT
│
├── .env                          # API keys for OpenAI, Claude, and Hugging Face
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
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

### **1. Data Collection**
Run the `dataCollection.ipynb` notebook to collect and preprocess data for price prediction. The notebook will generate:
- `train.pkl`: Preprocessed training dataset
- `test.pkl`: Preprocessed testing dataset

### **2. Traditional Machine Learning Models**
Use the `ProductPricerBaseModel.ipynb` notebook to train and evaluate traditional machine learning models like:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### **3. Frontier LLM Predictions**
Leverage `ProductPricerFrontierLLM.ipynb` to predict product prices using:
- GPT (OpenAI)
- Claude (Anthropic)

### **4. Fine-Tuning GPT**
Run `FineTuneOpenAIProductPricer.ipynb` to fine-tune GPT models for product price prediction using QLoRA. This step involves:
- Preparing the data in `fine_tune_train.jsonl` and `fine_tune_validation.jsonl`.
- Fine-tuning GPT-4O-mini with QLoRA to adapt it for the specific task.

---

## **Features**

- **Data Collection**: Gather and preprocess data for price prediction tasks.
- **Traditional ML Models**: Establish baseline models using classical machine learning techniques.
- **Frontier LLMs**: Utilize state-of-the-art models (Claude and GPT) for price prediction.
- **Fine-Tuning**: Customize GPT-4O-mini using QLoRA for improved task-specific performance.

---
