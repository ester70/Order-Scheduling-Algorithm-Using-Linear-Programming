#libraries
import pandas as pd
from datetime import datetime, timedelta
import pulp
from contextlib import redirect_stdout
import sys
import xlsxwriter
from pyairtable import Table
import requests
import json
import numpy as np
import pytz

#dynamo DB  libraries
import boto3
from botocore.exceptions import ClientError


# Airtable credentials
access_token = '...'
base_file_id = '...'
table_data = {
    'first_results': '...',
    'results': '...',
    'apartments': '...',
    'buildings': '...',
    'orders': '...',
    'blocked_dates': '...'
}

instance_id = '...'
region_name='...'
aws_access_key_id='...'
aws_secret_access_key='...'

# Fetching Airtable data and converting to DataFrames
def fetch_airtable_data():
    tables = {key: Table(access_token, base_file_id, table_data[key]).all() for key in table_data}
    dfs = {
        'buildings': pd.DataFrame.from_records([{'Building': r['fields'].get('Building'), 'ID': r['fields'].get('ID')} for r in tables['buildings']]),
        'apartments': pd.DataFrame.from_records([{'Building': r['fields'].get('Building'), 'Apartment': r['fields'].get('Apartment'), 'Beds': r['fields'].get('Beds'), 'ID': r['fields'].get('ID'), 'To_Schedule': r['fields'].get('To_Schedule')} for r in tables['apartments']]),
        'orders': pd.DataFrame.from_records([{'Order': r['fields'].get('Order'), 'Start_Date': r['fields'].get('Start_Date'), 'End_Date': r['fields'].get('End_Date'), 'Beds': r['fields'].get('Beds'), 'Price': r['fields'].get('Price'), 'Signed': r['fields'].get('Signed', False), 'ID': r['fields'].get('ID'),'To_Schedule': r['fields'].get('To_Schedule')} for r in tables['orders']]),
        'agreed_buildings': pd.DataFrame.from_records([{'Order': r['fields'].get('Order'), 'Agreed_building': r['fields'].get('Agreed_building')} for r in tables['orders']]),
        'blocked_dates': pd.DataFrame.from_records([{'Date': r['fields'].get('Date'), 'Building': r['fields'].get('Building'), 'Apartment': r['fields'].get('Apartment')} for r in tables['blocked_dates']]),
        'results': pd.DataFrame.from_records([{'Order': r['fields'].get('Order'), 'ID': r['fields'].get('ID'), 'Start_Date': r['fields'].get('Start_Date'), 'End_Date': r['fields'].get('End_Date'), 'Building': r['fields'].get('Building'), 'Apartment': r['fields'].get('Apartment')} for r in tables['results']]),
        'first_results': pd.DataFrame.from_records([{'Order': r['fields'].get('Order'), 'ID': r['fields'].get('ID'), 'Start_Date': r['fields'].get('Start_Date'), 'End_Date': r['fields'].get('End_Date'), 'Building': r['fields'].get('Building'), 'Apartment': r['fields'].get('Apartment')} for r in tables['first_results']])
    }
    return dfs

# Cleaning data
def clean_data(df, required_columns):
    return df.dropna(subset=required_columns) if not df.empty else df

# Converting to simple keys 
def convert_to_key(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else value

# Filter old orders
def remove_old_orders(orders_df):
    today = datetime.now().date()
    return orders_df[orders_df['Start_Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date()) > today]

# Delete future records from Airtable
def delete_future_records(results_df_to_delete, results_table_to_delete):
    today = datetime.now().date()
    
    for index, row in results_df_to_delete.iterrows():
        start_date = row.get('Start_Date', None)
        
            # Check if there is a valid start date
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            
            # If the date is greater than today, delete the record
            if start_date > today:
                record_id = row.get('ID', None)
                
            # If there is a record ID to delete
                if record_id:
                    try:
                        results_table_to_delete.delete(record_id)
                    except Exception as e:
                        pass 
# Date processing
def daterange(start_date, end_date):
    for n in range((end_date - start_date).days ):
        yield start_date + timedelta(n)

# Add blocked dates
def add_blocked_dates(results_df):
    if not results_df.empty:
        today = datetime.now().date()
        blocked_dates = pd.DataFrame(columns=['Date', 'Building', 'Apartment'])
        
        for index, row in results_df.iterrows():
            if row['Start_Date'] is not None and row['End_Date'] is not None:
                start_date = datetime.strptime(row['Start_Date'], '%Y-%m-%d').date()
                end_date = results_df['End_Date']

                if start_date <= today <= end_date:
                    for single_date in daterange(start_date, end_date):
                        blocked_dates = blocked_dates._append({
                            'Date': single_date.strftime("%Y-%m-%d"), 
                            'Building': row['Building'], 
                            'Apartment': row['Apartment']
                        }, ignore_index=True)
        return blocked_dates
    return pd.DataFrame(columns=['Date', 'Building', 'Apartment'])  #return DataFrame

# Define a function to stop the EC2 instance
def stop_ec2_instance(instance_id, region_name='us-east-1'):
    lambda_client = boto3.client('lambda',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key, region_name=region_name)     
    
# Activating the Lambda function to shut down the EC2 instance
    responselambda = lambda_client.invoke(
    FunctionName='shutdownec2', 
    InvocationType='RequestResponse',
    Payload=json.dumps({})  
    )
# Reading the results response
    response_payload = json.loads(responselambda['Payload'].read())

#Main process
dfs = fetch_airtable_data()

for key in dfs:
    dfs[key] = dfs[key].applymap(convert_to_key)

dfs['orders'] = clean_data(dfs['orders'], ['Order', 'Start_Date', 'End_Date', 'Price', 'Beds', 'ID'])
dfs['orders'] = remove_old_orders(dfs['orders'])
dfs['agreed_buildings'] = clean_data(dfs['agreed_buildings'],['Order','Agreed_building'])
orders_df = dfs['orders']
apartments_df = dfs['apartments']
agreed_buildings_df = dfs['agreed_buildings']
results_df = dfs['results']
first_results_df = dfs['first_results']
blocked_dates_df = dfs['blocked_dates']


# Blocked dates handling
blocked_dates = add_blocked_dates(results_df)
blocked_dates_df= pd.concat([blocked_dates_df, blocked_dates]).drop_duplicates(subset=['Building', 'Apartment', 'Date']).reset_index(drop=True)

# Delete future records from Airtable - call to function
results_table = Table(access_token, base_file_id, table_data['results'])
first_results_table = Table(access_token, base_file_id, table_data['first_results'])
delete_future_records(results_df, results_table)
delete_future_records(first_results_df,first_results_table)


#if orders not blank
if not orders_df.empty:
# preper data to model
    E = {row['ID']: datetime.strptime(row['Start_Date'], "%Y-%m-%d") for _, row in orders_df.iterrows()  if row['To_Schedule'] == True}
    L = {row['ID']: datetime.strptime(row['End_Date'], "%Y-%m-%d") for _, row in orders_df.iterrows()  if row['To_Schedule'] == True}
    B = {(convert_to_key(row['Building']), convert_to_key(row['ID'])): row['Beds'] for _, row in apartments_df.iterrows()  if row['To_Schedule'] == True}
    R = {row['ID']: row['Beds'] for _, row in orders_df.iterrows()}
    C = {row['ID']: row['Price'] for _, row in orders_df.iterrows()}
    contracts_signed = {row['ID']: row['Signed'] for _, row in orders_df.iterrows()}
    agreed_buildings = {row['ID']: row['agreed_buildings'] for _, row in agreed_buildings_df.iterrows()}

# model
    model = pulp.LpProblem("Scheduling_Problem", pulp.LpMaximize)

# variables
    x = pulp.LpVariable.dicts("x", [(i, b, d) for i in E for b, d in B], cat='Binary')
    y = pulp.LpVariable.dicts("y", [(i, b) for i in E for b in set(b for b, _ in B)], cat='Binary')

# the goal function
    bed_usage_cost = 100
    model += pulp.lpSum([C[i] * R[i] * ((L[i] - E[i]).days) * y[(i, b)] for i in E for b in {b for b, _ in B}])


# constraints
    constraints = []
# The total number of beds in the assigned apartments must be at least equal to the number of beds required for the reservation
    for i in E:
        for b in set(b for b, _ in B):
            constraint = pulp.lpSum([B[b2, d] * x[(i, b2, d)] for b2, d in B if b2 == b]) >= R[i] * y[(i, b)]
            model += constraint, f"Beds_for_order_{i}_building_{b}"
            constraints.append((f"Beds_for_order_{i}_building_{b}", constraint))

# No more than one reservation will be assigned to an apartment on any given date
    for b, d in B:
        for t in set(date for i in E for date in daterange(E[i], L[i])):
            constraint = pulp.lpSum([x[(i, b, d)] for i in E if E[i] <= t < L[i]]) <= 1
            model += constraint, f"Apartment_{b}_{d}_date_{t.strftime('%Y-%m-%d')}"
            constraints.append((f"Apartment_{b}_{d}_date_{t.strftime('%Y-%m-%d')}", constraint))

# A reservation will not be split between buildings
    for i in E:
        constraint = pulp.lpSum([y[(i, b)] for b in set(b for b, _ in B)]) <= 1
        model += constraint, f"One_building_per_order_{i}"
        constraints.append((f"One_building_per_order_{i}", constraint))

# Constraint: Link between x and y variables
    for i in E:
        for b, d in B:
            constraint = x[(i, b, d)] <= y[(i, b)]
            model += constraint, f"x_to_y_for_order_{i}_building_{b}_apartment_{d}"
            constraints.append((f"x_to_y_for_order_{i}_building_{b}_apartment_{d}", constraint))

# Constraint: Include all reservations for which a contract has been signed
    for i in E:
        if contracts_signed[i]:
            constraint = pulp.LpConstraint(
                e=pulp.lpSum([y[(i, b)] for b in set(b for b, _ in B)]),
                sense=pulp.LpConstraintEQ,
                rhs=1
            )
            model += constraint, f"Contract_signed_for_order_{i}"
            constraints.append((f"Contract_signed_for_order_{i}", constraint))

   # Constraint for blocked apartments
    for i in E: 
        for _, blocked_row in blocked_dates_df.iterrows():
            blocked_date = datetime.strptime(blocked_row['Date'], "%Y-%m-%d").date() 
            apartment_id = blocked_row['Apartment'] 
            building_id = blocked_row['Building']  
                 
# Checks if a blocked date falls within the order's date range, including the start date.
            if E[i].date() <= blocked_date < L[i].date():
                
                constraint = pulp.LpConstraint(
                    e=x[(i, building_id, apartment_id)],  
                    sense=pulp.LpConstraintEQ,
                    rhs=0
                )
                model += constraint, f"Blocked_date_{building_id}_{apartment_id}_{blocked_date.strftime('%Y-%m-%d')}_for_order_{i}"
            

# Constraint: If a building is agreed upon for an order, it will be assigned to the agreed building if it is assigned.
    for i in agreed_buildings:
        agreed_building = agreed_buildings[i]
        constraint = pulp.lpSum([y[(i, b)] for b in set(b for b, _ in B)]) == y[(i, agreed_building)]
        model += constraint, f"Agreed_building_for_order_{i}"
        constraints.append((f"Agreed_building_for_order_{i}", constraint))

   # Writing the constraints to a text file
    with open("constraints.txt", "w") as file:
        for name, constraint in constraints:
            file.write(f"{name}: {constraint}\n")
    

        
    solver = pulp.PULP_CBC_CMD(timeLimit=1000,warmStart=True,msg=True)
    with open("output.txt", "w") as f:
        with redirect_stdout(f):
            model.solve(solver)
            results = {}
            
            if pulp.LpStatus[model.status] == 'Optimal' or pulp.LpStatus[model.status] == 'Feasible':
                # results
                for i in E:
                    results[i] = [(b, d) for b, d in B if pulp.value(x[(i, b, d)]) == 1]

                
       # URL for Airtable API
    urlfirstresults = f"..."
    urlselectedresults = f"..."

    # Authorization headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    def insert_results_to_airtable(results, url):
        if not results:
            stop_ec2_instance(instance_id, region_name)  #Stop the EC2 instance if there are no results

        else:
            records = []
            for order, assignments in results.items():
                for building, apartment in assignments:
# Skip if there is no match
                    if building is None or apartment is None:
                        continue
# Create a record to add to Airtable
                    record_data = {
                        "fields": {
                            'Order': [order],   
                            'Building': [building],  
                            'Apartment': [apartment] 
                        }
                    }
                    records.append(record_data)
            # Sending the POST request to Airtable
            for i in range(0, len(records), 10):
                batch = records[i:i+10]
                data = {"records": batch}
                response = requests.post(url, headers=headers, data=json.dumps(data))


    # DynamoDB
    dynamodb = boto3.resource('dynamodb',aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key, region_name=region_name)
    table_dynamodb = dynamodb.Table('SchedulingResults')

     #DynamoDB
    def insert_results_to_dynamodb(results):
        
        for order, assignments in results.items():
            for building, apartment in assignments:
                if building is None or apartment is None:
                        continue

                try:
                        #DynamoDB
                        response = table_dynamodb.put_item(
                            Item={
                                'OrderID': order,
                                'BuildingID': building,
                                'ApartmentID': apartment
                            }
                        )
                except ClientError as e:
                        pass  
   
    insert_results_to_airtable(results,urlfirstresults)  # first results


# After completing the first assignment
# Use the output from the first assignment as the basis for E_selected and L_selected
    E_selected = {}
    L_selected = {}

    for i in E:
        for b in set(b for b, _ in B):
            if pulp.value(y[(i, b)]) == 1:
                E_selected[i] = E[i]
                L_selected[i] = L[i]


# Creating the model
    model_selected = pulp.LpProblem("Scheduling_Problem1", pulp.LpMaximize)
    x_selected = pulp.LpVariable.dicts("x", [(i, b, d) for i in E_selected for b, d in B], cat='Binary')
    y_selected = pulp.LpVariable.dicts("y", [(i, b) for i in E_selected for b in set(b for b, _ in B)], cat='Binary')


    w = pulp.LpVariable.dicts("w", [(i, b, d1, d2) for i in E_selected for b in set(b for b, _ in B) for (b2, d1) in B for (b3, d2) in B if b == b2 == b3 and d1 != d2], cat='Binary')


    apartment_numbers = {row['ID']: row['Apartment'] for _, row in apartments_df.iterrows()}  # Mapping apartment IDs to apartment numbers
    max_days_between_orders = 7 # Define a maximum distance of two weeks between reservations
    weights = {0: 40, 1: 40, 2: 15, 3: 20, 4: 25, 5: 30, 6: 35, 7: 40}# Assigning weights for days between reservations: positive values for larger gaps, negative values to discourage shorter gaps


# Objective function - maximize gaps between reservations if there is more than one day between them
    model_selected +=-pulp.lpSum([bed_usage_cost * B[b, d] * ((L_selected[i] - E_selected[i]).days ) * x_selected[(i, b, d)] for i in E_selected for b, d in B])+\
            0.1 * pulp.lpSum([
                 weights[min((E_selected[j] - L_selected[i]).days, max_days_between_orders)]
                for (b, d) in B  
                    for i, j in zip(
                    sorted([order for order in E_selected if x_selected[(order, b, d)] == 1], key=lambda x: E_selected[x])[1:],  # רק ההזמנות ששובצו לאותו בניין ודירה
                    sorted([order for order in L_selected if x_selected[(order, b, d)] == 1], key=lambda x: L_selected[x])  # רק ההזמנות ששובצו לאותו בניין ודירה
                    )
            if (E_selected[j] - L_selected[i]).days in weights

                ])+\
                0.01 * pulp.lpSum([
                w[(i, b, d1, d2)] * (1 / (1 + abs(apartment_numbers[d1] - apartment_numbers[d2])))
                for i in E_selected for b in set(b for b, _ in B) for (b2, d1) in B for (b3, d2) in B if b == b2 == b3 and d1 != d2
            ])

# List of constraints
    constraints_selected = []

# Setting constraints on the auxiliary variables w
    for i in E_selected:
        for b in set(b for b, _ in B):
            for (b2, d1) in B:
                for (b3, d2) in B:
                    if b == b2 == b3 and d1 != d2:
                        if (i, b, d1, d2) in w:
                            model_selected += w[(i, b, d1, d2)] <= x_selected[(i, b, d1)]
                            model_selected += w[(i, b, d1, d2)] <= x_selected[(i, b, d2)]
                            model_selected += w[(i, b, d1, d2)] >= x_selected[(i, b, d1)] + x_selected[(i, b, d2)] - 1
                        
# The total number of beds in the assigned building for a reservation must be at least equal to the number of beds required for the reservation
    for i in E_selected:
        for b in set(b for b, _ in B):
            constraint_selected = pulp.lpSum([B[b2, d] * x_selected[(i, b2, d)] for b2, d in B if b2 == b]) >= R[i] * y_selected[(i, b)]
            model_selected += constraint_selected, f"Beds_for_order_{i}_building_{b}"
            constraints_selected.append((f"Beds_for_order_{i}_building_{b}", constraint_selected))

# No more than one order will be assigned to an apartment on any given date
    for b, d in B:
        for t in set(date for i in E_selected for date in daterange(E_selected[i], L_selected[i])):
            constraint_selected = pulp.lpSum([x_selected[(i, b, d)] for i in E_selected if E_selected[i] <= t < L_selected[i]]) <= 1
            model_selected += constraint_selected, f"Apartment_{b}_{d}_date_{t.strftime('%Y-%m-%d')}"
            constraints_selected.append((f"Apartment_{b}_{d}_date_{t.strftime('%Y-%m-%d')}", constraint_selected))

# The reservation will not be split and must be assigned
    for i in E_selected:
        constraint_selected = pulp.lpSum([y_selected[(i, b)] for b in set(b for b, _ in B)]) == 1
        model_selected += constraint_selected, f"One_building_per_order_{i}"
        constraints_selected.append((f"One_building_per_order_{i}", constraint_selected))

# Link between x and y variables
    for i in E_selected:
        for b, d in B:
            constraint_selected = x_selected[(i, b, d)] <= y_selected[(i, b)]
            model_selected += constraint_selected, f"x_to_y_for_order_{i}_building_{b}_apartment_{d}"
            constraints_selected.append((f"x_to_y_for_order_{i}_building_{b}_apartment_{d}", constraint_selected))


# Constraint: An order will not be assigned to an apartment if there is an overlapping blocked date
    for i in E_selected:  
        for _, blocked_row in blocked_dates_df.iterrows():  
                blocked_date = datetime.strptime(blocked_row['Date'], "%Y-%m-%d").date()  
                apartment_id = blocked_row['Apartment'] 
                building_id = blocked_row['Building']  

                # Checks if a blocked date falls within the order's date range, including the start date.
                if E_selected[i].date() <= blocked_date < L_selected[i].date():   
                    constraint_selected = pulp.LpConstraint(
                        e=x_selected[(i, building_id, apartment_id)],  
                        sense=pulp.LpConstraintEQ,
                        rhs=0
                        )
                    model_selected += constraint_selected, f"Blocked_date_{building_id}_{apartment_id}_{blocked_date.strftime('%Y-%m-%d')}_for_order_{i}"
                    constraints_selected.append((f"Blocked_date_{building_id}", constraint_selected))

                
     
    # Constraint: If a building is agreed upon for an order, it will be assigned to the agreed building.
    for i in agreed_buildings:
        agreed_building = agreed_buildings[i]
        constraint_selected = pulp.LpConstraint(
            e=pulp.lpSum([y_selected[(i, agreed_building)]]),
            sense=pulp.LpConstraintEQ,
            rhs=1
            )
        model_selected += constraint_selected, f"Agreed_building_for_order_{i}"
        constraints_selected.append((f"Agreed_building_for_order_{i}", constraint_selected))

    # solve the problem 
    solver = pulp.PULP_CBC_CMD(timeLimit=3600,warmStart=True,msg=True)
    with open("output_selected.txt", "w") as f:
            with redirect_stdout(f):
                model_selected.solve(solver)
                results_selected = {}

                if pulp.LpStatus[model_selected.status] == 'Optimal' or pulp.LpStatus[model_selected.status] == 'Feasible':
                    # results
                    for i in E_selected:
                        results_selected[i] = [(b, d) for b, d in B if pulp.value(x_selected[(i, b, d)]) == 1]
                else:
                    stop_ec2_instance(instance_id, region_name)
    #main process
    if results_selected:
        insert_results_to_airtable(results_selected,urlselectedresults) 
        insert_results_to_dynamodb(results_selected)
    stop_ec2_instance(instance_id, region_name)


   