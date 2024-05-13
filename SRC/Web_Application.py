import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
import pandas as pd

with open('Artifacts/Models/Model_2.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature_names as a global variable
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = ['loan_int_rate', 'income_to_age_ratio', 'loan_amt_to_income_ratio', 'loan_percent_income', 
                        'emp_length_to_age_ratio', 'person_home_ownership_RENT', 'person_cred_hist_length', 
                        'loan_intent_HOMEIMPROVEMENT', 'person_home_ownership_MORTGAGE', 'loan_intent_MEDICAL', 
                        'person_home_ownership_OWN', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
                        'loan_intent_VENTURE', 'loan_intent_PERSONAL', 'person_home_ownership_OTHER']

app = dash.Dash(__name__)
server = app.server

# Define CSS styles
styles = {
    'container': {
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'fontFamily': 'Sans-Serif',
        'backgroundColor': '#F5F5F5'
    },
    'form': {
        'margin': '2em',
        'display': 'flex',
        'flex-direction' : 'column',
        'padding': '2em',
        'border': '1px solid #C0C0C0',
        'borderRadius': '5px',
        'backgroundColor': '#F9F9F9',
        'width': '60%'
    },
    'input': {
        'marginBottom': '0.5em',
        'padding-left': '5px'
    },
    'button': {
        'backgroundColor': '#008CBA',
        'color': 'white',
        'border': 'none',
        'padding': '15px 32px',
        'textAlign': 'center',
        'textDecoration': 'none',
        'display': 'inline-block',
        'fontSize': '16px',
        'margin': '4px 2px',
        'cursor': 'pointer',
        'borderRadius': '4px'
    }
}

# Define layout
app.layout = html.Div(style=styles['container'], children=[
    html.H1("Creditworthiness Assessment and Risk Analysis for Loan Default Prediction", style={'textAlign': 'center', 'color': '#7FDBFF'}),
    html.Div(style=styles['form'], children=[
        html.Label("Age:", style=styles['input']),
        dcc.Input(id='person_age', type='number', placeholder = 'Age'),
        html.Br(),
        html.Label("Income", style=styles['input']),
        dcc.Input(id = "person_income", type = 'number', placeholder = "Income"),
        html.Br(),
        html.Label("Home Ownership:", style=styles['input']),
        dcc.Dropdown(id="person_home_ownership", options = [
            {'label': 'Own', 'value':'OWN'},
            {'label':'Rent','value':'RENT'},
            {'label':'Mortgage','value': 'MORTGAGE'},
            {'label':'Other', 'value':'OTHER'}
        ], placeholder="Select Your Home Ownership"),
        html.Br(),
        html.Label("Employment Length in years", style=styles['input']),
        dcc.Input(id="person_emp_length",type='number',placeholder='Enter years in employment'),
        html.Br(),
        html.Label("Loan Intent", style=styles['input']),
        dcc.Dropdown(id='loan_intent', options = [
            {'label':'Personal','value':'Personal'},
            {'label':'Education','value':'EDUCATIONAL'},
            {'label':'Medical', 'value':'MEDICAL'},
            {'label':'Venture','value':'VENTURE'},
            {'label':'Home Improvements','value':'HOMEIMPROVEMENT'},
            {'label':'Debt Consolidation','value':'DEBTCONSOLIDATION'}],
        placeholder = 'Select Loan Intent'),
        html.Br(),
        html.Label("Loan Amount:", style=styles['input']),
        dcc.Input(id = 'loan_amnt', type='number', placeholder='Loan Amount'),
        html.Br(),
        html.Label("Interest Rate:", style=styles['input']),
        dcc.Input(id='loan_int_rate', type='number', placeholder = 'Interest Rate',step=0.01),
        html.Br(),
        html.Label("Loan Percent Income:", style=styles['input']),
        dcc.Input(id='loan_percent_income', type='number',placeholder = 'Loan percent Income',step=0.01),
        html.Br(),
        html.Label("Credit History Length in years:", style=styles['input']),
        dcc.Input(id='person_cred_hist_length',type='number',placeholder='Credit History length in years'),
        html.Br(),
        html.Label('Loan Status:', style={'font-weight': 'bold'}),
        html.Div(id='prediction-output', children='Fill in the form and press Enter'),
        html.Br(),
        html.Button("Submit", id = 'submit',n_clicks=0, style=styles['button'])
        ]),  
    ])

@app.callback(
    Output('prediction-output','children'),
    [Input('submit','n_clicks')],
    [State('person_age','value'),
    State('person_income','value'),
    State('person_home_ownership','value'),
    State('person_emp_length','value'),
    State('loan_intent','value'),
    State('loan_amnt','value'),
    State('loan_int_rate','value'),
    State('loan_percent_income','value'),
    State('person_cred_hist_length','value')])


def update_output(n_clicks, person_age, person_income, person_home_ownership, person_emp_length,
                loan_intent, loan_amnt, loan_int_rate, loan_percent_income,
                person_cred_hist_length):
    if n_clicks > 0:
        income_to_age_ratio = person_income / person_age
        loan_amt_to_income_ratio = loan_amnt / person_income
        emp_length_to_age_ratio = person_emp_length / person_age

        # Create a template DataFrame with all the necessary features
        template = pd.DataFrame(columns=feature_names, data=np.zeros((1, len(feature_names))))

        # Update the values based on the user's input
        template.at[0, 'loan_int_rate'] = loan_int_rate
        template.at[0, 'income_to_age_ratio'] = income_to_age_ratio
        template.at[0, 'loan_amt_to_income_ratio'] = loan_amt_to_income_ratio
        template.at[0, 'loan_percent_income'] = loan_percent_income
        template.at[0, 'emp_length_to_age_ratio'] = emp_length_to_age_ratio
        template.at[0, 'person_home_ownership_' + person_home_ownership] = 1
        template.at[0, 'person_cred_hist_length'] = person_cred_hist_length
        template.at[0, 'loan_intent_' + loan_intent] = 1

        # Make prediction
        prediction = model.predict(template)

        # Return the prediction result
        if prediction[0] == 1:
            return html.Div('Yes', style={'color': 'green'})
        else:
            return html.Div('No', style={'color': 'red'})
    else:
        return 'Fill in the form and press Enter'

if __name__ == "__main__":
    app.run_server(debug=False)
