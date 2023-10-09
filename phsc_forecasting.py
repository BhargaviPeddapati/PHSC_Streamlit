import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score
st.set_option('deprecation.showPyplotGlobalUse', False)


# Configure the page
st.set_page_config(page_title="Punjab Health Systems Corporation - Government of Punjab", page_icon="https://raw.githubusercontent.com/BhargaviPeddapati/PHSC_Streamlit/main/images/PC_LOGO.JPG", layout="wide")

logo = st.container()

with logo:
   
   col1, col2,col3,col4,col5 = st.columns([1,1,5,1,1])
   col2.image("https://raw.githubusercontent.com/BhargaviPeddapati/PHSC_Streamlit/main/images/PC_LOGO.JPG",  width=100)
   with col3:
     st.header('Punjab Health Systems Corporation - Government of Punjab')
     st.subheader('Demand forecasting of drugs and consumables')
   col4.image("https://raw.githubusercontent.com/BhargaviPeddapati/PHSC_Streamlit/main/images/ISB%20Logo.jpg",  width=100)
   st.markdown("""------""")
   

def load_data():
    
    data = pd.read_csv('https://raw.githubusercontent.com/BhargaviPeddapati/PHSC_Streamlit/main/datasets/DWH_df_with_AAC_details.csv')
    
    # Apply the filter where AAC Phase is Phase I
    data = data[data['AAC Phase'] == 'Phase I']
    
    # Convert the 'Date' column to a datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    
    # Reset the index if needed
    data.reset_index(drop=True, inplace=True)
    
    
    data = data[~((data['Date'].dt.month == 9) & (data['Date'].dt.year == 2023))]   
    
    data = data[~((data['Date'].dt.month == 8) & (data['Date'].dt.year == 2022))] 
    
    # Format the 'Date' column as yyyy-mm-dd
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # List of columns to convert to string
    columns_to_convert = ['Item Brand ID', 'Item ID', 'Issuing store ID', 'Receiving Store ID']
    
    # Convert selected columns to string
    data[columns_to_convert] = data[columns_to_convert].astype(str)
    
    # Remove .0 from values in the column
    data['Receiving Store ID'] = data['Receiving Store ID'].astype(str).str.rstrip('.0')
    
    reordered_columns = ['Date', 'Classification Name',	'Item Brand ID', 'Item ID','Item Name', 'Issuing store District', 'Issuing Store Type','Issuing store ID', 'Issuing store', 'Receiving store District', 'Receiving Store Type', 'Receiving Store ID', 'Receiving Store','Issue Qty']
    data = data[reordered_columns]
    
    return data


def data_weekly_division_type(filtered_table):
    # Convert 'Date' column to datetime format
    filtered_table['Date'] = pd.to_datetime(filtered_table['Date'])
    
    # Group by week and aggregate 'Issue Qty'
    weekly_data = filtered_table.groupby(filtered_table['Date'].dt.week)['Issue Qty'].sum()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data.index, weekly_data.values, marker='o', linestyle='-')
    plt.title('Weekly Analysis of Issue Qty')
    plt.xlabel('Week Number')
    plt.ylabel('Total Issue Qty')
    plt.grid(True)
    st.pyplot()
    
    # Group by 'Item Name' and week, then sum the 'Issue Qty' for each week
    weekly_data = filtered_table.groupby(['Item Name', filtered_table['Date'].dt.week])['Issue Qty'].sum().unstack().fillna(0)
    
    # Select the week to display (user input)
    selected_week = st.selectbox('**Select Week:**', weekly_data.columns)
    
    # Filter data for the selected week
    data_for_week = weekly_data[selected_week]
    data_for_week = data_for_week[data_for_week > 0]
    
    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(data_for_week, labels=data_for_week.index, autopct='%1.1f%%')
    ax.set_title(f'Pie Chart for Week {selected_week}')
    
    # Display the pie chart in Streamlit
    st.pyplot(fig)

def data_monthly_division_type(filtered_table):       
    
    # Create two columns for the two rows
    col1, col2 = st.columns(2)     
  
    filtered_table['Date'] = pd.to_datetime(filtered_table['Date'])
    
    # Extract month and year from the 'Date' column
    filtered_table['Month'] = filtered_table['Date'].dt.month
    filtered_table['Year'] = filtered_table['Date'].dt.year
    
    
    # Group the data by 'Item Name', 'Year', and 'Month' and sum the 'Issue Qty'
    monthly_item_data = filtered_table.groupby(['Item Name', 'Year', 'Month'])['Issue Qty'].sum().reset_index()
    
    
   # Loop through each selected item name
    for selected_item_name in filtered_table['Item Name'].unique():
        # Filter the data based on the selected item name
        filtered_data = monthly_item_data[monthly_item_data['Item Name'] == selected_item_name]
        
        # Pivot the filtered data to prepare for a stacked area plot
        pivot_data = filtered_data.pivot_table(index='Month', columns='Year', values='Issue Qty', fill_value=0)
        
        
        with col1:  
            st.subheader(f'Issue Qty Variation for {selected_item_name} Over Months')
            st.dataframe(pivot_data, height=466) 
            
        with col2:                  
            # Plot demand variation for the selected item name over months using a stacked area plot
            plt.figure(figsize=(12, 8))
            sns.set_palette("husl", len(pivot_data.columns))
            sns.set(style="whitegrid")
            plt.stackplot(pivot_data.index, pivot_data.values.T, labels=pivot_data.columns)
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.xlabel('Month')
            plt.ylabel('Total Issue Qty')
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            plt.tight_layout()
            st.pyplot()
            
    with col1:  
        st.dataframe(filtered_table)
       
        
    with col2:
        # Display the table
        # Plot demand by Item Name (top N items for better visualization)        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=filtered_table, x='Item Name', y='Issue Qty')
        #plt.xticks(rotation=90)
        plt.xticks(rotation=90, horizontalalignment='right')
        plt.tight_layout()
        st.pyplot()  
    
    # Iterate through the list of drugs and plot seasonal decomposition
    for selected_drug in filtered_table['Item Name'].unique():
        
        drug_data = filtered_table[filtered_table['Item Name'] == selected_drug]       
        
        # Plot demand over time
        st.subheader(f'Demand Over Time for {selected_drug}')
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=drug_data, x='Date', y='Issue Qty')
        st.pyplot()
        
        
        st.subheader(f'Seasonal Decomposition for {selected_drug}')
        
        result = seasonal_decompose(drug_data['Issue Qty'], model='additive', period=1)
        
        plt.figure(figsize=(12, 8))
        plt.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0.5})
        
        plt.subplot(411)
        plt.plot(result.observed)
        plt.title('Observed')
        
        plt.subplot(412)
        plt.plot(result.trend)
        plt.title('Trend')
        
        plt.subplot(413)
        plt.plot(result.seasonal)
        plt.title('Seasonal')
        
        plt.subplot(414)
        plt.plot(result.resid)
        plt.title('Residual')
        
        # Apply tight layout
        plt.tight_layout()
        
        st.pyplot()
        
        
def call_division_method(filtered_table): 
    
    
    st.header('Weekly Analysis') 
    
    st.markdown("""------""")
    
    weekly_df = import_data_weekly()
    
    st.subheader('Dataframe')
    
    st.dataframe(weekly_df, height=250)     
    
    data_weekly_division_type(filtered_table)  
          
    
    st.markdown("""------""")
    
    
    st.header('Monthly Analysis')   
    
    st.markdown("""------""")
    
    monthly_df = import_data_monthly()      
    
    st.subheader('Dataframe')
    
    st.dataframe(monthly_df, height=250)         
    
    data_monthly_division_type(filtered_table)
    
    st.markdown("""------""")

def find_indexes_and_names_of_medication_names(medication_list, medication_names_to_find):
    results = []
    for name in medication_names_to_find:
        try:
            index = medication_list.index(name)
            results.insert(index,name)
        except ValueError:
            # Handle the case where the medication name is not found in the list
            results.append((None, name))
    return results   

def handle_option_change(data, option):
    
    if option == "Select from Item Name":
        all_option = ["All"]        
        # Get distinct values from the "Item Name" column
        distinct_item_name = data["Item Name"].unique()        
        distinct_item_name = all_option + list(distinct_item_name)
        medication_names_to_find = ['Cough Syrup: Each 5ml to contain Dextromethorphan HCL/Hbr 10mg, Phenylephrine 5mg and CPM 2mg','Multivitamin Tab', 'Cefixime Tab 200 mg', 'Levo Cetrizine Tab  5 mg']
        default_selection = find_indexes_and_names_of_medication_names(distinct_item_name, medication_names_to_find)
        selected_drugs = st.multiselect("**Select Item Names:**", distinct_item_name, default=default_selection)
        filtered_table = data[data['Item Name'].isin(selected_drugs)]
        top_items_demand_table = filtered_table.groupby('Item Name')['Issue Qty'].sum().reset_index()
        top_items_demand_table.columns = ['Item Name', 'Total Issue Qty']
        top_items_demand_table = top_items_demand_table.sort_values(by='Total Issue Qty', ascending=False)
        top_items_demand_table = top_items_demand_table.head(5)

    
    elif option == "Top N Drugs":
        top_n_items = st.number_input("**Enter the top value:**", min_value=1, step=1, value=5)
        st.write(f'Demand by Top {top_n_items} Item Names')
        top_items_demand_table = data.groupby('Item Name')['Issue Qty'].sum().reset_index()
        top_items_demand_table.columns = ['Item Name', 'Total Issue Qty']
        top_items_demand_table = top_items_demand_table.sort_values(by='Total Issue Qty', ascending=False)
        st.write(top_items_demand_table)
        top_items_demand_table = top_items_demand_table.head(top_n_items)
        filtered_table = data[data['Item Name'].isin(top_items_demand_table["Item Name"].unique())]

    
    filtered_table.to_csv("C:/Users/ADMIN/Desktop/data science/ISB/PHSC/AAC/DWH_df_with_AAC_details_filtered1.csv")
    
    # Add a horizontal line to separate attribute summaries
    st.write('---') 
    
    call_division_method(filtered_table)
    

# Create the tab row
def visualize_data(data):
    
    # Define Streamlit app
    st.title('Exploratory Data Analysis')
    
    data = data.drop(columns=['Unnamed: 0'])
       
    # Add a horizontal line to separate attribute summaries
    st.write('---')
    st.subheader('Summary Statistics for Each Attribute')
   
    sunmarry_df = data[[ 'Item Name', 'Date', 'Receiving Store']]  
    
    sunmarry_df['Date'] = pd.to_datetime(sunmarry_df['Date'])
    
    
   # Specify the number of columns you want to display the statistics side by side
    num_columns = 3
    column_width = 12 // num_columns  # Divide the available width into equal parts
    
    # Calculate the number of rows needed based on the number of attributes and columns
    num_attributes = len(sunmarry_df.columns)
    num_rows = (num_attributes + num_columns - 1) // num_columns
    
    # Create Streamlit columns
    columns = st.columns(num_columns)
    
    # Summary Statistics for Each Attribute
    for row in range(num_rows):
        for col in range(num_columns):
            idx = row * num_columns + col
            if idx >= num_attributes:
                break
            column = sunmarry_df.columns[idx]
            with columns[col]:
                st.write(f"**{column}**:")
                st.write(f"Number of Non-null Entries: {sunmarry_df[column].count()}")
                unique_values = sunmarry_df[column].unique()
                unique_values_frequency = sunmarry_df[column].value_counts()
                # Create a new DataFrame from the unique values and their frequencies
                unique_values_df = pd.DataFrame({column: unique_values_frequency.index, 'Frequency': unique_values_frequency.values})

                st.write(f"Number of Unique Entries: {len(unique_values)}")
                st.dataframe(unique_values_df, height=400,width=800)
                 
                if sunmarry_df[column].dtype == 'datetime64[ns]':
                    # Handle Date columns differently                        
                    st.write(f"Minimum Date: {sunmarry_df[column].min()}")
                    st.write(f"Maximum Date: {sunmarry_df[column].max()}")
                elif sunmarry_df[column].dtype in ['float64', 'int64']:
                    
                    # For numeric columns
                    
                    # Count of zeros for numeric columns
                    count_zeros = ((sunmarry_df[column] == 0) | (sunmarry_df[column].isna())).sum()
                    st.write(f"Count of Zeros and Empties: {count_zeros}")
                    
                    st.write(f"Mean: {sunmarry_df[column].mean()}")
                    st.write(f"Median: {sunmarry_df[column].median()}")
                    st.write(f"Standard Deviation: {sunmarry_df[column].std()}")
                    st.write(f"Minimum: {sunmarry_df[column].min()}")
                    st.write(f"Maximum: {sunmarry_df[column].max()}")
    
    # Add a horizontal line to separate attribute summaries
    st.write('---') 
    
    
    # Dropdown to select the option
    option = st.selectbox("**Choose an option:**", ["Select from Item Name", "Top N Drugs"], index=1)

    
    handle_option_change(data, option)
    

def load_tab1():
    # Create a container for the tab row
    tab1_data_container = st.container()
    
    # Create the tab row
    with tab1_data_container:
        # Load the data
        data = load_data()
        
        all_option = ["All"]
    
        # Get distinct values from the "Classification Name" column
        distinct_classification_name = data["Classification Name"].unique()
        
        # Add "All" to the list of options
        distinct_classification_name = all_option + list(distinct_classification_name)        
        
        # Get distinct values from the "Item Name" column
        distinct_item_name = data["Item Name"].unique()
        
        distinct_item_name = all_option + list(distinct_item_name)
        
        
        # Get distinct values from the "Issuing store District" column
        distinct_issuing_store_district = data["Issuing store District"].unique()
        
        # Add "All" to the list of options
        distinct_issuing_store_district = all_option + list(distinct_issuing_store_district)
        
        # Get distinct values from the "Issuing store" column
        distinct_issuing_store = data["Issuing store"].unique()
        
        # Add "All" to the list of options
        distinct_issuing_store = all_option + list(distinct_issuing_store)
    
        
        # Get distinct values from the "Receiving store District" column
        distinct_receiving_store_district = data["Receiving store District"].unique()
        
        
        # Add "All" to the list of options
        distinct_receiving_store_district = all_option + list(distinct_receiving_store_district)
        
        # Get distinct values from the "Receiving Store" column
        distinct_receiving_store = data["Receiving Store"].unique()
        
        # Add "All" to the list of options
        distinct_receiving_store = all_option + list(distinct_receiving_store)
        
        # Create two columns for the two rows
        col1, col2 = st.columns(2)
        
        
        # Create dropdowns in the first row
        with col1:
            # Create a dropdown select box
            selected_classification_name = st.selectbox("**Classification Name:**", distinct_classification_name)
        
        with col2:
            # Create a dropdown select box
            selected_item_name = st.selectbox("**Item Name:**", distinct_item_name)
            
            
        # Create dropdowns in the second row
        with col1:
            # Create a dropdown select box
            selected_issuing_store_district = st.selectbox("**Issuing store District:**", distinct_issuing_store_district)
        
        with col2:
            # Create a dropdown select box
            selected_issuing_store = st.selectbox("**Issuing store:**", distinct_issuing_store)
    
        # Create dropdowns in the third row
        with col1:
           # Create a dropdown select box
           selected_receiving_store_district = st.selectbox("**Receiving store District:**", distinct_receiving_store_district)
        
        with col2:
            # Create a dropdown select box
            selected_receiving_store = st.selectbox("**Receiving Store:**", distinct_receiving_store)  
        
        
        with col1:
            st.write("**Classification Name:** ", selected_classification_name)
            
        with col2:
            st.write("**Item Name:** ", selected_item_name)
            
        with col1:
            st.write("**Issuing store District:**" , selected_issuing_store_district)
            
        with col2:
            st.write("**Issuing store:**", selected_issuing_store)
            
        with col1:
            st.write("**Receiving store District:**" , selected_receiving_store_district)
            
        with col2:
            st.write("**Receiving Store:**", selected_receiving_store) 
            
            
        # Filter the DataFrame based on selections
        filtered_df = data[
            ((data["Classification Name"] == selected_classification_name) | (selected_classification_name == 'All')) &
            ((data["Item Name"] == selected_item_name ) | (selected_item_name == 'All')) &
            ((data["Issuing store District"] == selected_issuing_store_district) | (selected_issuing_store_district == 'All'))  &
            ((data["Issuing store"] == selected_issuing_store) | (selected_issuing_store == 'All')) &
            ((data["Receiving store District"] == selected_receiving_store_district) | (selected_receiving_store_district == 'All')) &
            ((data["Receiving Store"] == selected_receiving_store) | (selected_receiving_store == 'All')) 
        ]
        
        
        st.subheader('Summary Statistics for Each Attribute')
       
        st.write(f"Total Records: {filtered_df.shape[0]}")
        st.write(f"Total Attributes: {filtered_df.shape[1]}")
        
        
        sunmarry_df = filtered_df[[ 'Item Name', 'Date', 'Receiving Store']]  
        
        sunmarry_df['Date'] = pd.to_datetime(sunmarry_df['Date'])
        
        
       # Specify the number of columns you want to display the statistics side by side
        num_columns = 3
        column_width = 12 // num_columns  # Divide the available width into equal parts
        
        # Calculate the number of rows needed based on the number of attributes and columns
        num_attributes = len(sunmarry_df.columns)
        num_rows = (num_attributes + num_columns - 1) // num_columns
        
        # Create Streamlit columns
        columns = st.columns(num_columns)
        
        # Summary Statistics for Each Attribute
        for row in range(num_rows):
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx >= num_attributes:
                    break
                column = sunmarry_df.columns[idx]
                with columns[col]:
                    st.write(f"**{column}**:")
                    st.write(f"Number of Non-null Entries: {sunmarry_df[column].count()}")
                    unique_values = sunmarry_df[column].unique()
                    unique_values_frequency = sunmarry_df[column].value_counts()
                    # Create a new DataFrame from the unique values and their frequencies
                    unique_values_df = pd.DataFrame({column: unique_values_frequency.index, 'Frequency': unique_values_frequency.values})

                    st.write(f"Number of Unique Entries: {len(unique_values)}")
                    st.dataframe(unique_values_df, height=400,width=800)
                     
                    if sunmarry_df[column].dtype == 'datetime64[ns]':
                        # Handle Date columns differently                        
                        st.write(f"Minimum Date: {sunmarry_df[column].min()}")
                        st.write(f"Maximum Date: {sunmarry_df[column].max()}")
                    elif sunmarry_df[column].dtype in ['float64', 'int64']:
                        
                        # For numeric columns
                        
                        # Count of zeros for numeric columns
                        count_zeros = ((sunmarry_df[column] == 0) | (sunmarry_df[column].isna())).sum()
                        st.write(f"Count of Zeros and Empties: {count_zeros}")
                        
                        st.write(f"Mean: {sunmarry_df[column].mean()}")
                        st.write(f"Median: {sunmarry_df[column].median()}")
                        st.write(f"Standard Deviation: {sunmarry_df[column].std()}")
                        st.write(f"Minimum: {sunmarry_df[column].min()}")
                        st.write(f"Maximum: {sunmarry_df[column].max()}")
                   

                    
        
        # Add a horizontal line to separate attribute summaries
        st.write('---')
                
        # Display the filtered DataFrame
        st.write(filtered_df)
    
        shape = filtered_df.shape
        rows = shape[0]
        columns = shape[1]
        
        st.write("**Rows**",rows, "   **Columns**",columns)
        
        
        # Add a button for submission
        if st.button("Save Data"):
            filtered_df.to_csv("C:/Users/ADMIN/Desktop/data science/ISB/PHSC/AAC/DWH_df_with_AAC_details_filtered.csv")
        
def load_tab2():
    filtered_df = pd.read_csv('C:/Users/ADMIN/Desktop/data science/ISB/PHSC/AAC/DWH_df_with_AAC_details_filtered.csv')
    
    visualize_data(filtered_df)

# Tab 3 - Section Code


## Preparations of Time Series


def import_data_weekly():
    AAC_receivables = pd.read_csv("C:/Users/ADMIN/Desktop/data science/ISB/PHSC/AAC/DWH_df_with_AAC_details_filtered1.csv")

    AAC_receivables['Date'] = pd.to_datetime(AAC_receivables['Date'])
    AAC_receivables['Week'] = AAC_receivables['Date'].dt.week

    AAC_receivables['Period'] = AAC_receivables['Week'].astype(str).str.zfill(2)
    df = pd.pivot_table(data=AAC_receivables, values='Issue Qty', index = ['Item Name'], columns = 'Period', aggfunc = 'sum', fill_value = 0)
    return df

def import_data_bi_weekly():
    AAC_receivables = import_data_weekly() # Get the number of columns in the DataFrame
    num_columns = AAC_receivables.shape[1]
    
    # Create new columns by summing adjacent columns
    for i in range(0, num_columns, 2):
        new_col_name = f'{i}-{i+1}'
        AAC_receivables[new_col_name] = AAC_receivables.iloc[:, i] + AAC_receivables.iloc[:, i+1]

    # Delete the original columns
    AAC_receivables.drop(AAC_receivables.columns[:num_columns], axis=1, inplace=True)
    return AAC_receivables


def import_data_monthly():
    receivables = pd.read_csv("C:/Users/ADMIN/Desktop/data science/ISB/PHSC/AAC/DWH_df_with_AAC_details_filtered1.csv")

    receivables['Date'] = pd.to_datetime(receivables['Date'])
    receivables['Month'] = receivables['Date'].dt.strftime('%Y-%m')

    df = pd.pivot_table(data=receivables, values='Issue Qty', index=['Item Name'], columns='Month', aggfunc='sum', fill_value=0)

    print(df.shape)
    return df


def fastMoving(df):

  # Calculate the threshold for non-zero values
  threshold = df.shape[1] * 0.9

  # Select rows where at least threshold number of columns are non-zero
  selected_rows = df[(df != 0).sum(axis=1) >= threshold]

  # Print the selected rows
  return selected_rows


def MediumMoving(df):

  # Calculate the threshold for non-zero values
  lower_threshold = df.shape[1] * 0.5
  upper_threshold = df.shape[1] * 0.9
  # Select rows where at least threshold number of columns are non-zero
  selected_rows = df[((df != 0).sum(axis=1) >= lower_threshold) & ((df != 0).sum(axis=1) < upper_threshold)]

  # Print the selected rows
  return selected_rows


### Preparing Data for Training and Test Data sets

## Train Test Splitting Data

def datasets(df, x_len = 5, y_len = 1, test_loops = 5):
    D = df.values
    rows, periods = D.shape

    ## Training set creation
    loops = periods + 1 - x_len - y_len
    train = []

    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])

    train = np.vstack(train)


    X_train, Y_train = np.split(train,[-y_len], axis = 1)

    ## Test Set Creation

    if(test_loops > 0):
        X_train, X_test = np.split(X_train, [-rows*test_loops], axis = 0)
        Y_train, Y_test = np.split(Y_train, [-rows*test_loops], axis = 0)

    else:
        X_test = D[:,-x_len:]
        Y_test = np.full((X_test.shape[0],y_len),np.nan)


    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test

## Define KPI for ML
kpis = []

def kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = ''):

    # MAE, RMSE  are interms of percentages.
    df = pd.DataFrame(columns = ['MAE','RMSE','Bias', 'R-squared'], index=['Train','Test'])

    df.index.name = name

    df.loc['Train','MAE'] = 100 * np.mean(abs(Y_train - Y_train_pred))/np.mean(Y_train)
    df.loc['Train','RMSE'] = 100 * np.sqrt(np.mean((Y_train - Y_train_pred)**2))/np.mean(Y_train)
    df.loc['Train','Bias'] = 100 * np.mean((Y_train - Y_train_pred))/np.mean(Y_train)
    df.loc['Train','R-squared']  = 100 * r2_score(Y_train, Y_train_pred)

    df.loc['Test','MAE'] = 100 * np.mean(abs(Y_test - Y_test_pred))/np.mean(Y_test)
    df.loc['Test','RMSE'] = 100 * np.sqrt(np.mean((Y_test - Y_test_pred)**2))/np.mean(Y_test)
    df.loc['Test','Bias'] = 100 * np.mean((Y_test - Y_test_pred))/np.mean(Y_test)
    df.loc['Test','R-squared'] = 100 * r2_score(Y_test, Y_test_pred)
 
    # Create DataFrames with a specific length
    num_rows = 10  # Change this to the desired length
    df_train = pd.DataFrame({
        'Actual (Train)': np.round(Y_train[:num_rows],2),
        'Predicted (Train)':np. round(Y_train_pred[:num_rows],2),
        'Percentage Difference (Train)': np.round( ((Y_train_pred[:num_rows] - Y_train[:num_rows]) / Y_train[:num_rows]) * 100 , 2)
    })

    df_test = pd.DataFrame({
        'Actual (Test)': np.round(Y_test[:num_rows],2),
        'Predicted (Test)': np.round( Y_test_pred[:num_rows],2),
        'Percentage Difference (Test)': np.round( ((Y_test_pred[:num_rows] - Y_test[:num_rows]) / Y_test[:num_rows]) * 100 , 2)
    })
    
    # Create two columns for the two rows
    col1, col2 = st.columns(2)
    
    # Print the results
    st.subheader(f"Results for {name} Model:")
    
    with col1:
        st.subheader("Train Data:")
        st.dataframe(df_train)
        
    with col2:
        st.subheader("Test Data:")
        st.dataframe(df_test)

    df = df.astype(float).round(1)
    st.dataframe(df)
    kpis.append(df)



def model_LinearRegression(X_train, Y_train, X_test, Y_test, df):
    
    from sklearn.linear_model import LinearRegression
    ## Linear Regression suits for stable short term forecasts which is our baseline
    reg = LinearRegression()
    reg = reg.fit(X_train, Y_train)
    
    Y_train_pred = reg.predict(X_train)
    Y_test_pred = reg.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'Regression')
    
    X_train, Y_train, X_test, Y_test = datasets(df, x_len = 5, y_len = 1 , test_loops = 0)
    forecast_df = pd.DataFrame(data =  reg.predict(X_test), index = df.index)

    forecast_df = forecast_df.rename(columns={0:'Forecast'})
    forecast_df = forecast_df.round(1)
    st.dataframe(forecast_df)
    
    df_with_forecast = pd.merge(df, forecast_df, on='Item Name')
    
    st.dataframe(df_with_forecast.head())



def model_decision_tree_regression(X_train, Y_train, X_test, Y_test, df):
    
    ## K FOLD CROSS VALIDATION RANDOM SEARCH ALGORITHM
    max_depth = list(range(3,20)) + [None]
    min_samples_split = range(10,40)
    min_samples_leaf = range(10,20)
    
    param_dist = { 'max_depth' : max_depth,
                   'min_samples_split' : min_samples_split,
                   'min_samples_leaf' : min_samples_leaf
                 }
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.tree import DecisionTreeRegressor
    
    ## MAE Optimized
    tree = DecisionTreeRegressor()
    tree_cv = RandomizedSearchCV(tree, param_dist, n_jobs = -1, cv = 10, verbose = 1, n_iter = 100, scoring ='neg_mean_absolute_error')
    
    tree_cv.fit(X_train, Y_train)
    
    #print('Tuned regression Tree Parameters: ', tree_cv.best_params_)
    
    Y_train_pred = tree_cv.predict(X_train)
    Y_test_pred = tree_cv.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Optimized Tree')



def model_mae(model, X,Y):
    Y_pred = model.predict(X)
    mae = np.mean(np.abs(Y-Y_pred))/np.mean(Y)
    return mae


## APPROACH 3 : Holdout Set - 2nd Test Set

def datasets_holdout(df, x_len = 5, y_len = 1, test_loops = 5, holdout_loops = 5):
    D = df.values
    rows, periods = D.shape

    # Training set creations
    train_loops = periods + 1 - x_len - y_len - test_loops
    train = []

    for col in range(train_loops):
        train.append(D[:, col:col+x_len+y_len])

    train = np.vstack(train)

    X_train, Y_train = np.split(train, [-y_len], axis = 1)

    ## Holdout set creation

    if holdout_loops > 0:
        X_train, X_holdout = np.split(X_train, [-rows*holdout_loops], axis = 0)
        Y_train, Y_holdout = np.split(Y_train, [-rows*holdout_loops], axis = 0)

    else :
        X_holdout, Y_holdout = np.array([]),np.array([])

    ## est Set Creation

    if test_loops > 0:
        X_train, X_test = np.split(X_train, [-rows*test_loops], axis = 0)
        Y_train, Y_test = np.split(Y_train, [-rows*test_loops], axis = 0)
    else : ## No test set: X_test is used to generate the future forecast
        X_test = D[:, -x_len:]
        Y_test = np.full((X_test.shape[0], y_len), np.nan)

    ## Formatting required for scikit-learn

    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()
        Y_holdout = Y_holdout.ravel()


    return X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test

def model_random_forest_regression(X_train, Y_train, X_test, Y_test, df):
    
    ##  Optimize Random Forest - WISDOM OF CROWD
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    
    max_depth = list(range(3,20)) + [None]
    min_samples_split = range(10,40)
    min_samples_leaf = range(10,20)
    
    max_features = range(3,8)
    
    bootstrap = [True]
    
    max_samples = [0.7, 0.8,0.9, 0.95, 1]
    
    max_features = range(3,8)
    
    param_dist = { 'max_depth': max_depth,
                   'min_samples_split' : min_samples_split,
                   'min_samples_leaf' : min_samples_leaf,
                   'max_features' : max_features,
                   'bootstrap' : bootstrap,
                   'max_samples' : max_samples}
    
    ## Estimators is for Trees
    forest = RandomForestRegressor(n_jobs = 1 , n_estimators = 10)
    
    forest_cv = RandomizedSearchCV(forest, param_dist, cv=6, n_jobs = -1, verbose = 2, n_iter = 400, scoring = 'neg_mean_absolute_error')
    
    forest_cv.fit(X_train, Y_train)
    
    Y_train_pred = forest_cv.predict(X_train)
    Y_test_pred = forest_cv.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'Forest Optimized')
    
    
    st.header('Forestx200')    
    
    forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, **forest_cv.best_params_)
    forest = forest.fit(X_train, Y_train)
    
    Y_train_pred = forest.predict(X_train)
    Y_test_pred = forest.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'Forestx200')
    
    ## Understanding the feature importance in model accuracy. That is which week demand is impacting the forecasts

    cols = X_train.shape[1]
    features = [f'W-{cols-col}' for col in range(cols)]
    
    data = forest.feature_importances_.reshape(-1,1)
    
    imp = pd.DataFrame(data=data, index=features, columns=['Forest'])
    imp.plot(kind='bar')
    
    st.header('Extra Trees Regressor')
    
    from sklearn.ensemble import ExtraTreesRegressor

    ETR = ExtraTreesRegressor(n_jobs = -1, n_estimators = 200, min_samples_split = 18, min_samples_leaf = 10, max_samples=0.95, max_features = 7, max_depth = 17, bootstrap=True)
    
    ETR.fit(X_train, Y_train)
    
    Y_train_pred = ETR.predict(X_train)
    Y_test_pred = ETR.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'ETR')
    
    
    st.header('Optimize Extreme Random Forest')    
    
    ##  Optimize Extreme Random Forest

    max_depth = list(range(3,20)) + [None]
    min_samples_split = range(10,40)
    min_samples_leaf = range(10,20)
    
    max_features = range(2,9)
    
    bootstrap = [True]
    
    max_samples = [0.7, 0.8,0.9, 0.95, 1]
    
    max_features = range(3,8)
    
    param_dist = { 'max_depth': max_depth,
                   'min_samples_split' : min_samples_split,
                   'min_samples_leaf' : min_samples_leaf,
                   'max_features' : max_features,
                   'bootstrap' : bootstrap,
                   'max_samples' : max_samples}
    
    ## Estimators is for Trees
    ETR = ExtraTreesRegressor(n_jobs = 1 , n_estimators = 10)
    
    ETR_cv = RandomizedSearchCV(ETR, param_dist, cv=5, n_jobs = -1, verbose = 2, n_iter = 400, scoring = 'neg_mean_absolute_error')
    
    ETR_cv.fit(X_train, Y_train)
    
    Y_train_pred = ETR_cv.predict(X_train)
    Y_test_pred = ETR_cv.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'ETR Optimized')
    
    
    ## Use this as first GO To Model as ETR is better and as fast as Forest with lesser training time about 30%
    ETR = ExtraTreesRegressor(n_estimators = 200, n_jobs = -1, **ETR_cv.best_params_).fit(X_train, Y_train)
    
    Y_train_pred = ETR.predict(X_train)
    Y_test_pred = ETR.predict(X_test)
    
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name = 'ETRx200')
    
    
    forest_features = { 'n_jobs':-1, 'n_estimators':200, 'min_samples_split':1983, 'min_samples_leaf':44, 'max_samples':0.95, 'max_features':6,'max_depth':83, 'bootstrap':True}
    forest = RandomForestRegressor(**forest_features)
    
    ETR_features = { 'n_jobs':-1, 'n_estimators':200, 'min_samples_split':1991, 'min_samples_leaf':56, 'max_samples':0.9, 'max_features':4,'max_depth':96, 'bootstrap':True}
    
    ETR = ExtraTreesRegressor(**ETR_features)
    
    models = [('forest',forest), ('ETR', ETR)]
    
    n_trees = range(1,9,2)

    results = []
    
    for x_len in n_trees:
        X_train, Y_train, X_test, Y_test = datasets(df, x_len = x_len)
    
        for name , model in models:
    
            if np.all(X_train == 0):
              continue
            model.fit(X_train, Y_train)
            mae_train = model_mae(model, X_train, Y_train)
            mae_test = model_mae(model, X_test, Y_test)
    
            results.append([name+'Train', mae_train,x_len])
            results.append([name+'Test', mae_test,x_len])
    
    data = pd.DataFrame(results, columns = ['Model','MAE%', 'Number of Trees'])
    data = data.set_index(['Number of Trees','Model']).stack().unstack('Model')
    
    data.index = data.index.droplevel(level=1)
    
    data.index.name = 'Number of Trees'
    
    data.plot(color =['orange']*2 + ['black']*2, style=['-','--']*2)
    
    st.dataframe(data.idxmin())
    
    ## Using K-fold validations to arrive at right feature numbers

    from sklearn.model_selection import KFold
    
    results = []
    n_weeks = range(1,17,2)
    
    for x_len in n_trees:
        X_train, Y_train, X_test, Y_test = datasets(df, x_len = x_len)
        for name, model in models:
            mae_kfold_train = []
            mae_kfold_val = []
    
            for train_index, val_index in KFold(n_splits = 5).split(X_train):
                X_train_kfold, X_val_kfold = X_train[train_index], X_train[val_index]
                Y_train_kfold, Y_val_kfold = Y_train[train_index], Y_train[val_index]
    
                model.fit(X_train_kfold, Y_train_kfold)
    
                mae_train = model_mae(model, X_train_kfold, Y_train_kfold)
                mae_val = model_mae(model,X_val_kfold,Y_val_kfold )
    
                mae_kfold_train.append(mae_train)
                mae_kfold_val.append(mae_val)
    
            results.append([name+' Val',np.mean(mae_kfold_val),x_len])
            results.append([name+' Train', np.mean(mae_kfold_train), x_len])
    
            model.fit(X_train, Y_train)
    
            mae_test = model_mae(model, X_test, Y_test)
    
            results.append([name+' Test', mae_test, x_len])
            
            
    data = pd.DataFrame(results, columns = ['Model', 'MAE%', 'Number of Trees'])
    data = data.set_index(['Number of Trees', 'Model']).stack().unstack('Model')
    
    data.index = data.index.droplevel(level=1)
    
    data.index.name = 'Number of Trees'
    
    data.plot(color=['orange'] * 3+ ['black'] * 3, style=['-','--',':']*2)
    
    st.dataframe(data.idxmin())
    
    
    results = []
    n_weeks = range(1,5,2)
    for x_len in n_weeks:
        X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test = datasets_holdout(df, x_len = x_len, holdout_loops = 5)
        if(len(X_train) == 0):
          continue
        for name, model in models:
            model.fit(X_train, Y_train)
    
            mae_train = model_mae(model, X_train, Y_train)
            mae_holdout = model_mae(model, X_holdout, Y_holdout)
            mae_test = model_mae(model, X_test, Y_test)
    
            results.append([name+' Train', mae_train, x_len])
            results.append([name+' Test', mae_test, x_len])
            results.append([name+' Holdout', mae_holdout, x_len])
    
    
def model_XGBRegressor(X_train, Y_train, X_test, Y_test, df):
    from xgboost.sklearn import XGBRegressor
    X_train, Y_train, X_test, Y_test = datasets(df,x_len = 5, y_len = 1, test_loops = 5)
    
    XGB = XGBRegressor(n_jobs = 1, max_depth = 50, n_estimators = 100, learning_rate = 0.2)
    
    XGB = XGB.fit(X_train, Y_train)
    
    import xgboost as xgb

    XGB.get_booster().feature_names = [f'W{x-5}' for x in range(5)]
    
    xgb.plot_importance(XGB, importance_type = 'total_gain', show_values = False)
    
    
    ## Using Multi Regressor to forecast multiple future periods

    from sklearn.multioutput import MultiOutputRegressor
    X_train, Y_train, X_test, Y_test = datasets(df,x_len = 5, y_len = 3, test_loops = 0)
    
    XGB = XGBRegressor(n_jobs=1, max_depth = 4 , n_estimators = 100, learning_rate = 0.2)
    
    multi = MultiOutputRegressor(XGB, n_jobs = -1)
    
    multi.fit(X_train, Y_train)
    
    ## Forecasting future periods

    X_train, Y_train, X_test, Y_test = datasets(df,x_len = 7, y_len = 2, test_loops = 0)
    
    XGB = XGBRegressor(n_jobs=1, max_depth = 4 , n_estimators = 100, learning_rate = 0.2)
    
    multi = MultiOutputRegressor(XGB, n_jobs = -1)
    
    multi.fit(X_train, Y_train)
    forecast = pd.DataFrame(data=multi.predict(X_test),   index=df.index)
    
    forecast = forecast.round(1)
    
    st.dataframe(forecast)
    
    ## XG Boosting using Evaluation Set
    from xgboost.sklearn import XGBRegressor
    import xgboost as xgb
    
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=5, y_len=1, test_loops=5)
    
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
    
    XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=2000, learning_rate=0.01)
    XGB = XGB.fit(x_train, y_train, early_stopping_rounds=100, verbose=True, eval_set=[(x_val, y_val)], eval_metric='mae')
    
    XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=2000, learning_rate=0.01)
    XGB = XGB.fit(x_train, y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='mae')
    
    st.write(f'Best iteration: {XGB.get_booster().best_iteration}')
    st.write(f'Best score: {XGB.get_booster().best_score}')
    
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=5, y_len=1, test_loops=5)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
    
    params = {'max_depth': [5,6,7,8,10,11],
            'learning_rate': [0.005,0.01,0.025,0.05,0.1,0.15],
            'colsample_bynode' : [0.5,0.6,0.7,0.8,0.9,1.0],#max_features
            'colsample_bylevel': [0.8,0.9,1.0],
            'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
            'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7],#max_samples
            'min_child_weight': [5,10,15,20,25],#min_samples_leaf
            'reg_alpha': [1,5,10,20,50],
            'reg_lambda': [0.01,0.05,0.1,0.5,1],
            'n_estimators':[1000]}
    
    fit_params = {'early_stopping_rounds':25,
                'eval_set':[(x_val, y_val)],
                'eval_metric':'mae',
                'verbose':False}
    
    from sklearn.model_selection import RandomizedSearchCV
    
    XGB = XGBRegressor(n_jobs=1)
    XGB_cv = RandomizedSearchCV(XGB, params, cv=5, n_jobs=-1, verbose=1, n_iter=1000, scoring='neg_mean_absolute_error')
    XGB_cv.fit(x_train, y_train,**fit_params)
    print('Tuned XGBoost Parameters:',XGB_cv.best_params_)
    
    best_params = XGB_cv.best_params_
    XGB = XGBRegressor(n_jobs=-1, **best_params)
    XGB = XGB.fit(x_train, y_train, **fit_params)
    print(f'Best iteration: {XGB.get_booster().best_iteration}')
    print(f'Best score: {XGB.get_booster().best_score}')
    Y_train_pred = XGB.predict(X_train)
    Y_test_pred = XGB.predict(X_test)
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')
    
    ## Exporting Code to Tuned Dataset
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=5, y_len=1, test_loops=0)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
    
    XGB = XGBRegressor(n_jobs=-1, **best_params)
    XGB = XGB.fit(x_train, y_train, **fit_params)
    Y_train_pred = XGB.predict(X_train)
    Y_test_pred = XGB.predict(X_test)
            
    forecast = pd.DataFrame(data= XGB.predict(X_test), index=df.index)

    forecast = forecast.rename(columns = {0:'FW1'})
    forecast = forecast.round(1)
    
    st.dataframe(forecast)
    
    df_with_forecast = pd.merge(df, forecast, on = 'Item Name')
    # Calculate the percentage of zeros in each row
    df_with_forecast['Issued_Percentage'] = round((df_with_forecast != 0).sum(axis=1) / df_with_forecast.shape[1] * 100,1)
    st.dataframe(df_with_forecast)
    
    kpis_df = pd.concat(kpis, keys=[df.index.name for df in kpis])
    
    st.dataframe(kpis_df)
    
def modelling(df):
    
    st.dataframe(df, height=250) 
    
    # The training set is 5 weeks period for forecasting one week ahead that is 6th week. The training set are  rolling window period of 1 week.
    X_train, Y_train, X_test, Y_test = datasets(df,x_len = 5, y_len = 1, test_loops = 5)
    
    st.header('Linear Regression')
    
    
    model_LinearRegression(X_train, Y_train, X_test, Y_test, df)
    
    
    st.header('Optimized Tree')
    
    model_random_forest_regression(X_train, Y_train, X_test, Y_test, df)
    
    
    st.header('XGBRegressor')
    
    model_XGBRegressor(X_train, Y_train, X_test, Y_test, df)

def load_tab3():
    # Create a container for the tab row
    tab3_data_container = st.container()
    
    # Create the tab row
    with tab3_data_container:
        
        # Forecasting
        
        #st.header('Dataframe for Monthly Data')
        
        #df = import_data_monthly()
        
        #modelling(df)
        
        st.header('Dataframe for weekly Data')
        
        df = import_data_bi_weekly()
        
        modelling(df)
        
        
# Create a container for the tab row
tabs_container = st.container()

# Create the tab row
with tabs_container:
    # Create three columns for the tabs
    col1, col2, col3  = st.columns(3)

    # Create a tab button in each column
    tab1_selected = col1.button("Data Overview & Preparation")
    tab2_selected = col2.button("Exploratory Data Analysis")
    tab3_selected = col3.button("Forecasting")
    
    # Add content to the selected tab
    if tab3_selected:
        load_tab3()
    elif tab2_selected:
        load_tab2()
    else:
        load_tab1()
    
    
   