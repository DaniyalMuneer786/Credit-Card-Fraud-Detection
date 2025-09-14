# 💳🔍 Credit-Card-Fraud-Detection
🔹 About the Project
- This project helps detect fraudulent credit card transactions using machine learning.
- The system uses a smart model (Random Forest) to check if a transaction is real or fake.
- It is useful for banks, customers, and businesses to prevent financial fraud.
---

🔹 Dataset
- The dataset contains credit card transaction details like:
- Credit Card Number
- Merchant
- Category
- Amount
- Location (City, State, Latitude, Longitude)
- Transaction Date and Number
---

🔹 Data Balance Technique
- The original data was unbalanced, so we used SMOTE to make it balanced and improve model performance.
---

🔹 Model Used
- Model: Random Forest Classifier
- Reason: Works well for classification tasks like fraud detection
- Accuracy: ~99% (very high, no overfitting or underfitting)
---

🔹 Tech Stack
- Frontend: HTML, CSS
- Backend: Python (Jupyter Notebook)
- Framework: Flask (connects frontend with backend)
---

🔹 How It Works
- User fills in transaction details via a web form.
- Data is sent to the Flask backend.
- The Random Forest model checks the transaction.
- The result page shows whether the transaction is Fraud or Not Fraud.
---

🔹 How to Run the Project
- Open the project folder.
- Run the Flask app (press Ctrl+F5).
- A web form will open.
- Enter transaction details and submit.
- Get the result on the screen.
---

🔹 Folder Rules (Important)
- ✔ All HTML files must be in the templates folder.
- ✔ All images and CSS files must be in the static folder.
- ✔ CSS files should be inside: static/css
---

🔹 Why this Project is Useful?
- Helps prevent financial loss due to fraud.
- Easy to use for customers and bank employees.
- Very high accuracy (99%).
- Quick and automated fraud detection.
---

🙏 Thank You for Checking Out This Project!
---
