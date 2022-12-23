import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#establishing key metrics
lead_orders_22 = 2259
total_leads_22 = 56468
search_conv_rate = .0558
close_rate = 0.04000495856060069
fb_conv_rate = 0.015146803126899535

#establishing months for ml model inputs
August = [1,0,0,0,0,0,0,0,0,0,0]
December = [0,1,0,0,0,0,0,0,0,0,0]
February = [0,0,1,0,0,0,0,0,0,0,0]
January = [0,0,0,1,0,0,0,0,0,0,0]
July = [0,0,0,0,1,0,0,0,0,0,0]
June = [0,0,0,0,0,1,0,0,0,0,0]
March = [0,0,0,0,0,0,1,0,0,0,0]
May = [0,0,0,0,0,0,0,1,0,0,0]
November = [0,0,0,0,0,0,0,0,1,0,0]
October = [0,0,0,0,0,0,0,0,0,1,0]
September = [0,0,0,0,0,0,0,0,0,0,1]
April = [0,0,0,0,0,0,0,0,0,0,0]



#Order Goals - our goal is here to increase Wholesale orders by 10% for 2023. 
#Note - according to Google Analytics, ~44% of conversions (leads) came from paid sessions (Facebook & Search Combined)
#Note - according

Goal_Orders_23 = (lead_orders_22 * 1.1) / 12

#knowing our close rate, we can calculate how many leads are needed to hit the orders goal

Goal_Leads_23 = Goal_Orders_23 / close_rate

#defining function to calculate goal given user input for non paid leads

def lead_goal_line(non_paid_leads):
    
    goal_leads = Goal_Leads_23 - non_paid_leads
    
    return goal_leads

#defining function to calculate orders goal given user input for non paid leads
def orders_goal_line(non_paid_leads):
    
    goal_leads = Goal_Leads_23 - non_paid_leads
    
    goal_orders = goal_leads * close_rate
    
    return goal_orders

#defining 2 functions to read in trained models
def load_fb_model():

    model = pickle.load(open('fb_model.pkl', 'rb'))
    
    return model

def load_search_model():
    
    model = pickle.load(open('ep_model.pkl', 'rb'))
    
    return model

#assigning both models once loaded
fb_model = load_fb_model()
search_model = load_search_model()

#defining function to run ep/search model
def run_model(cost, month):
    aug = month[0]
    dec = month[1]
    feb = month[2]
    jan = month[3]
    jul = month[4]
    jun = month[5]
    mar = month[6]
    may = month[7]
    nov = month[8]
    oct_ = month[9]
    sep = month[10]
    
    pred = search_model.predict([[cost, aug, dec, feb, jan, jul, jun, mar, may, nov, oct_, sep]])
    
    return(pred)


def show_predict_page():
    st.title("Ad Spend Impact Prediction")
    
    st.write("""### Model is based on 2022 performance""")
    
    fb_daily_budget = []
    search_daily_budget = []
    
    fb_daily = st.number_input('Facebook Daily Budget', step = 250)
    google = st.number_input('Google Daily Budget', step = 250)
    bing =  st.number_input('Bing Daily Budget', step = 250)
    month = st.selectbox('Select Month', ('January','February','March','April','May','June','July','August','September','October','November','December'))
    non_paid_leads = st.number_input('Expected Organic Leads', step = 250)
    
    if month == 'January':
        month = January
        
    if month == 'February':
        month = February
        
    if month == 'March':
        month = March
        
    if month == 'April':
        month = April
        
    if month == 'May':
        month = May
        
    if month == 'June':
        month = June
        
    if month == 'July':
        month = July
        
    if month == 'August':
        month = August
        
    if month == 'September':
        month = September
        
    if month == 'October':
        month = October
        
    if month == 'November':
        month = November
        
    
    fb_daily_budget.append(fb_daily)
    search_daily_budget.append(google + bing)
    
    ok = st.button("Predict")
    if ok:
    
    
        daily_fb_budgets = []
        daily_search_budgets = []

        for fb, search in zip(fb_daily_budget, search_daily_budget):
            daily_fb_budgets.append(fb *.7)
            daily_fb_budgets.append(fb *.8)
            daily_fb_budgets.append(fb *.9)
            daily_fb_budgets.append(fb)
            daily_fb_budgets.append(fb *1.1)
            daily_fb_budgets.append(fb *1.2)
            daily_fb_budgets.append(fb *1.3)
    
            daily_search_budgets.append(search *.7)
            daily_search_budgets.append(search *.8)
            daily_search_budgets.append(search *.9)
            daily_search_budgets.append(search)
            daily_search_budgets.append(search *1.1)
            daily_search_budgets.append(search *1.2)
            daily_search_budgets.append(search *1.3)
    
    

            #establishing empty lists to hold calculations for charting
            fb_order_predictions = []
            search_order_predictions = []
            total_order_predictions = []
            fb_lead_predictions = []
            search_lead_predictions = []
            total_lead_predictions = []
            fb_traffic_predictions = []
            search_traffic_predictions = []
            total_traffic_predictions = []
            monthly_fb_budget = []
            monthly_search_budget = []
            total_monthly_budget = []


            for fb, search in zip(fb_daily_budget, search_daily_budget):
                daily_fb_budgets.append(fb *.7)
                daily_fb_budgets.append(fb *.8)
                daily_fb_budgets.append(fb *.9)
                daily_fb_budgets.append(fb)
                daily_fb_budgets.append(fb *1.1)
                daily_fb_budgets.append(fb *1.2)
                daily_fb_budgets.append(fb *1.3)
    
                daily_search_budgets.append(round(search *.7))
                daily_search_budgets.append(round(search *.8))
                daily_search_budgets.append(round(search *.9))
                daily_search_budgets.append(search)
                daily_search_budgets.append(round(search *1.1))
                daily_search_budgets.append(round(search *1.2))
                daily_search_budgets.append(round(search *1.3))
    
    

            #establishing empty lists to hold calculations for charting
            fb_order_predictions = []
            search_order_predictions = []
            total_order_predictions = []
            fb_lead_predictions = []
            search_lead_predictions = []
            total_lead_predictions = []
            fb_traffic_predictions = []
            search_traffic_predictions = []
            total_traffic_predictions = []
            monthly_fb_budget = []
            monthly_search_budget = []
            total_monthly_budget = []


            for fb_spend, search_spend in zip(daily_fb_budgets, daily_search_budgets):
    
                #creating monthly budgets from daily budgets
                fb_monthly_spend = fb_spend * 30
                monthly_fb_budget.append(fb_monthly_spend)
                search_monthly_spend = search_spend * 30
                monthly_search_budget.append(search_monthly_spend)
                total_monthly_budget.append(round(search_monthly_spend) + round(fb_monthly_spend))
    
    
                #running the models on the fb & search budgets and storing results in list objects for graphing later
    
                fb_traffic_pred = fb_model.predict([[fb_spend]])[0][0]
                fb_traffic_pred = fb_traffic_pred * 30
                fb_traffic_predictions.append(fb_traffic_pred)
    
    
                search_traffic_pred = run_model(search_spend, month)[0][0]
                search_traffic_pred = search_traffic_pred * 30
                search_traffic_predictions.append(search_traffic_pred)
                total_traffic_pred = fb_traffic_pred + search_traffic_pred
                total_traffic_predictions.append(total_traffic_pred)
    
                #taking traffic predictions from models and calculating lead/conversions by multiplying traffic x conversion rate on each platform
                fb_lead_prediction = fb_traffic_pred * fb_conv_rate
                fb_lead_predictions.append(fb_lead_prediction)
                search_lead_prediction = search_traffic_pred * search_conv_rate
                search_lead_predictions.append(search_lead_prediction)
                total_lead_preds = fb_lead_prediction + search_lead_prediction
                total_lead_predictions.append(total_lead_preds)

                #taking lead predictions and multiplying by close rate to get order predictions
                fb_order_prediction = fb_lead_prediction * close_rate
                fb_order_predictions.append(fb_order_prediction)
                search_order_prediction = search_lead_prediction * close_rate
                search_order_predictions.append(search_order_prediction)
                total_order_preds = fb_order_prediction + search_order_prediction
                total_order_predictions.append(total_order_preds)  
                
            data_tuples = list(zip(total_monthly_budget, total_lead_predictions))
            df1 = pd.DataFrame(data_tuples, columns = ['Monthly_Budget', 'Monthly_Leads'])
            df1['Monthly_Budget'] = '$' + df1['Monthly_Budget'].astype(str)


                
        
            
            fig, ax = plt.subplots(figsize=(25,15))
            chart_traffic = ax.bar(df1['Monthly_Budget'], total_traffic_predictions, label = 'Google & Bing', color = 'indianred')
            chart_traffic_fb = plt.bar(df1['Monthly_Budget'], fb_traffic_predictions, label = 'Facebook', color = 'cornflowerblue')
            plt.title('Predicted Paid Site Traffic', size = 35)
            plt.ylabel('Paid Site Traffic', size = 30)
            plt.xlabel('Monthly Ad Spend', size = 30)
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=30)
            plt.legend(fontsize=25, loc='upper left')
            for bar in chart_traffic:
                height = bar.get_height()
                label_x_pos = bar.get_x() + bar.get_width() / 2
                plt.text(label_x_pos, height, s=f'{round(height):,}', ha = 'center', va = 'bottom', size=25)
                
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(25,15))
            
            chart_leads = plt.bar(df1['Monthly_Budget'], total_lead_predictions, label = 'Google & Bing', color = 'indianred')
            chart_leads_fb = plt.bar(df1['Monthly_Budget'], fb_lead_predictions, label = 'Facebook', color = 'cornflowerblue')
            plt.title('Predicted Paid Leads Generated', size = 35)
            plt.ylabel('Paid Leads', size = 30)
            plt.xlabel('Monthly Ad Spend', size = 30)
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=30)
            for bar in chart_leads:
                height = bar.get_height()
                label_x_pos = bar.get_x() + bar.get_width() / 2
                plt.text(label_x_pos, height, s=f'{round(height):,}', ha = 'center', va = 'bottom', size=25) 
    
            #for bar in chart_leads_fb:
            #height = bar.get_height()
            #label_x_pos = bar.get_x() + bar.get_width() / 2
            #plt.text(label_x_pos, height, s=f'{round(height):,}', ha = 'center', va = 'top', size=12)
            plt.axhline(y = lead_goal_line(non_paid_leads), color = 'grey', linestyle = 'dotted', linewidth=6, alpha = .5)
            plt.text(7, lead_goal_line(non_paid_leads),'Lead Goal',ha='left', va='center', size =25, alpha = .9)
            plt.legend(fontsize=25, loc='upper left')
            
            
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(25,15))
            
            chart_orders = plt.bar(df1['Monthly_Budget'], total_order_predictions, label = 'Google & Bing', color = 'indianred')
            chart_orders_fb = plt.bar(df1['Monthly_Budget'], fb_order_predictions, label = 'Facebook', color = 'cornflowerblue')
            plt.title('Predicted Orders from Paid Leads', size = 35)
            plt.ylabel('Orders from Paid Leads', size = 30)
            plt.xlabel('Monthly Ad Spend', size = 30)
            plt.tick_params(axis='x', labelsize=30)
            plt.tick_params(axis='y', labelsize=30)
            for bar in chart_orders:
                height = bar.get_height()
                label_x_pos = bar.get_x() + bar.get_width() / 2
                plt.text(label_x_pos, height, s=f'{round(height):,}', ha = 'center', va = 'bottom', size=25)
    
                #for bar in chart_orders_fb:
                #height = bar.get_height()
                #label_x_pos = bar.get_x() + bar.get_width() / 2
                #plt.text(label_x_pos, height, s=f'{round(height):,}', ha = 'center', va = 'top', size=12)
            plt.axhline(y = orders_goal_line(non_paid_leads), color = 'grey', linestyle = 'dotted', linewidth=6, alpha = .5)
            plt.text(7, orders_goal_line(non_paid_leads),'Orders Goal',ha='left', va='center', size =25, alpha = .9)
            plt.legend(fontsize=25, loc='upper left')
                
            st.pyplot(fig)
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
            
if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password")
        return False
elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
else:
        # Password correct.
        return True
    
if check_password():
    st.button("Click me")    
    
show_predict_page()
