
# Order Scheduling Algorithm Using Linear Programming

This project implements a **scheduling algorithm** to assign customer orders to apartments in buildings efficiently, using **Linear Programming** (via `pulp.CBC`) and integrating various technologies such as **AWS Lambda**, **EC2**, and **Airtable** for data handling and system management.

---

## 🚀 Project Overview

The primary goal of this project is to optimize order assignments to apartments while considering:
- Apartment availability and capacity.
- Customer preferences and agreed-upon constraints (e.g., specific buildings).
- Blocked dates for apartments and buildings.
- Maximizing revenue and minimizing inefficiencies (e.g., underutilized beds).

The project leverages **Python** and is designed to run on an **AWS EC2 instance**, with automated shutdown and restart capabilities managed by **AWS Lambda**.

---

## 📋 Features

- **Optimization:** Uses **Linear Programming** (`pulp.CBC`) to solve scheduling problems efficiently.
- **Dynamic Data Management:** 
  - Fetches input data from **Airtable**.
  - Updates results back to Airtable after execution.
  - Stores results in **AWS DynamoDB** for long-term data retention.
- **Blocked Dates Handling:** Prevents scheduling conflicts by respecting predefined blocked dates.
- **Agreed Constraints:** Ensures specific customer orders are assigned to agreed-upon buildings.
- **AWS Integration:**
  - **EC2**: Runs the scheduling algorithm.
  - **Lambda**: Manages automatic EC2 shutdown.
- **Customizable Objective Function:** Maximizes revenue, optimizes gaps between orders, and minimizes disruptions.

---

## 🛠️ Technologies Used

- **Python**: Main programming language for the project.
- **Pulp**: Linear Programming solver (`PULP_CBC_CMD`) for optimization.
- **Airtable**: Used for input and output data management.
- **AWS EC2**: Hosts the scheduling algorithm.
- **AWS Lambda**: Automates EC2 instance shutdown.
- **AWS DynamoDB**: Stores processed results for persistence.
- **Libraries Used**:
  - `pandas`, `numpy`, `pulp`, `pyairtable`, `requests`, `boto3`, `datetime`, `pytz`.

---

## 🗂️ Repository Structure

```
.
├── src/
│   ├── main.py             # Core scheduling algorithm
│   ├── data_processing.py  # Data cleaning and formatting functions
│   ├── airtable_integration.py # Fetching and updating Airtable data
│   ├── aws_integration.py  # Functions for AWS Lambda and EC2 management
│   └── dynamodb_integration.py # Inserting results into AWS DynamoDB
├── data/
│   ├── input_data.csv      # Example input data
│   └── output_results.csv  # Generated output results
├── constraints.txt         # List of constraints applied to the model
├── output.txt              # Solver output for first optimization
├── output_selected.txt     # Solver output for secondary optimization
└── README.md               # Project description and usage guide
```

---

## 📊 Example Input and Output

### Input Example (Airtable):

#### **Orders Table:**
| Order ID | Start Date  | End Date    | Beds | Price Per Night | Signed | Agreed Building |
|----------|-------------|-------------|------|-----------------|--------|-----------------|
| 1        | 2024-01-15  | 2024-01-20  | 4    | 150             | Yes    | Building A      |
| 2        | 2024-02-01  | 2024-02-05  | 6    | 120             | No     | Building B      |

#### **Apartments Table:**
| Apartment ID | Building     | Beds Available | To Schedule |
|--------------|--------------|----------------|-------------|
| Apt 101      | Building A   | 4              | True        |
| Apt 102      | Building A   | 6              | True        |
| Apt 201      | Building B   | 5              | True        |
| Apt 202      | Building B   | 6              | False       |

#### **Blocked Dates Table:**
| Blocked Date | Building     | Apartment |
|--------------|--------------|-----------|
| 2024-01-16   | Building A   | Apt 101   |
| 2024-01-17   | Building B   | Apt 201   |

---

### Output Example:

#### **Results Table:**
| Order ID | Assigned Building | Assigned Apartment | Start Date  | End Date    | Total Price |
|----------|-------------------|--------------------|-------------|-------------|-------------|
| 1        | Building A        | Apt 102           | 2024-01-15  | 2024-01-20  | $3000       |
| 2        | Building B        | Apt 203           | 2024-02-01  | 2024-02-05  | $3600       |

---

## 📦 Setup and Execution

### Prerequisites:
1. **AWS Credentials:** Ensure you have valid `aws_access_key_id` and `aws_secret_access_key`.
2. **Airtable API Key:** Provide your Airtable API access token and base/table IDs.

### Steps to Run:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ester70/Order-Scheduling-Algorithm-Using-Linear-Programming.git
   ```
2. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up AWS Lambda:**
   - Deploy the `shutdownec2` function in AWS Lambda to handle EC2 shutdowns.
   - Ensure appropriate permissions are configured for Lambda to interact with EC2.

4. **Run the main script:**
   ```bash
   python src/main.py
   ```

---

## 📝 How It Works

1. **Data Fetching:**
   - Data is pulled from Airtable tables (e.g., orders, apartments, buildings, blocked dates).
   - Data is cleaned and formatted into `pandas` DataFrames.

2. **First Optimization:**
   - The Linear Programming model (`pulp.CBC`) assigns orders to apartments based on constraints such as:
     - Bed capacity.
     - Blocked dates.
     - Agreed building preferences.
   - Results are stored in Airtable and DynamoDB.

3. **Secondary Optimization:**
   - After the first assignment, a secondary optimization is performed to maximize gaps between reservations and minimize disruptions.

4. **EC2 Management:**
   - AWS Lambda shuts down the EC2 instance after execution, saving costs.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).

---

Happy Scheduling! 😊
