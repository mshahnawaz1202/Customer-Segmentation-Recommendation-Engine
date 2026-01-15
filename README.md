
# Customer Segmentation & Recommendation Engine

## Overview

This project is a Streamlit-based application that performs end-to-end customer
segmentation and product recommendation using transactional retail data.

### Key Features

- Data loading and cleaning
- RFM analysis (Recency, Frequency, Monetary)
- Average Order Value calculation
- K-Means clustering with elbow method
- Customer personas per cluster
- Product recommendation engine

---

## Project Structure

```

Customer Segmentation & Recommendation Engine/
├── app.py
├── requirements.txt
├── README.md
└── data.csv

````

---
## Tech Used

- Numpy
- Pandas
- Matplotlib
- Sckit-learn
- Streamlit
  
## Getting Started

### Requirements

- Python 3.10 or higher
- pip

### Installation

```bash
cd "d:/Customer Segmentation & Recommendation Engine"
pip install -r requirements.txt
````

### Run the Application

```bash
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

## Dataset


```
data.csv
```

Expected columns:

* InvoiceNo
* StockCode
* Description
* Quantity
* InvoiceDate
* UnitPrice
* CustomerID
* Country

---

## Notes

* This project uses a single fixed CSV file
* Designed for academic  use
* Easily extendable with dashboards and reports

---

## Author

**Muhammad Shah Nawaz**
Customer Analytics | Data Science | Machine Learning





