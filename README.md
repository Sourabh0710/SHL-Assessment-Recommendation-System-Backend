#  SHL Assessment Recommendation System – Backend

This repository contains the **backend API** for the **SHL Assessment Recommendation System**, developed as part of the **SHL Internship Assessment**.

The backend is implemented using **FastAPI** and exposes a **RESTful API** that accepts a job description or requirement text and returns the most relevant **SHL assessments** in **JSON format**, using **semantic similarity techniques**.

---

##  Live Deployment

###  Backend API (FastAPI)

**Base URL:**  

https://shl-assessment-recommendation-system-462r.onrender.com/
---

###  Recommendation Endpoint (Main)

POST /recommend

**Example:**  
https://shl-assessment-recommendation-system-462r.onrender.com/recommend

---

### API Documentation (Swagger)

https://shl-assessment-recommendation-system-462r.onrender.com/docs

---

## Frontend (Streamlit Web App)

The backend is integrated with a **Streamlit-based frontend** for easy interaction and demonstration.

**Live Web App:**  
https://shl-assessment-recommendation-system-25.streamlit.app/

### Users can:
- Enter job descriptions or requirement text
- Choose the number of recommendations
- View clean, tabular recommendation results
- Download recommendations as a CSV file

### Frontend Repository (Main Repo)
https://github.com/Sourabh0710/SHL-Assessment-Recommendation-System

---

##  How It Works

1. The user submits a **job description** via the API or frontend
2. The input text is converted into a **TF-IDF vector**
3. Semantic similarity is computed against SHL assessment metadata
4. Assessments are ranked using **cosine similarity**
5. The most relevant assessments are returned as **JSON**

The system only returns **meaningful matches**, ensuring high relevance and avoiding noisy recommendations.

---

##  API Usage

###  Request

**Endpoint:**  
POST /recommend

**Example Request Body:**

{
  "text": "Python Developer with strong problem solving skills",
  "max_results": 10
}
###  Response (JSON)

[
  {
    "assessment_name": "Python (New)",
    "test_type": "K",
    "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
    "score": 0.61
  }
]

## Project Structure

### backend
- api.py: FastAPI application
- recommender.py: Recommendation engine
- shl_catalog.csv: SHL assessment dataset
- requirements.txt: Backend dependencies
- README.md: Documentation
## Technologies Used
Python 3.11
FastAPI – REST API framework
scikit-learn – TF-IDF vectorization & cosine similarity
pandas – Data handling
Uvicorn – ASGI server
Render – Backend deployment
Streamlit – Frontend user interface

## Key Features

Public REST API returning JSON responses
Semantic similarity-based assessment recommendations
Defensive error handling (no runtime crashes)
Clean, evaluator-friendly responses
Separate backend and frontend architecture
Interactive Swagger documentation included
